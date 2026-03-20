"""
MQTT Async Client
-----------------

Async wrapper around paho-mqtt using asyncio.Queue as a bridge.

Features
--------
- Publish control action  (QoS 1)
- Subscribe state / telemetry  (QoS 0)
- Emergency stop  (QoS 2, retained)
- Auto-reconnect with exponential back-off

Design
------
paho-mqtt runs its own background thread.
All callbacks push messages into asyncio.Queue so the rest of
the codebase can await them without touching threads directly.

Usage
-----
    async with MQTTAsyncClient(config) as mqtt:
        await mqtt.publish_control({"pwm": 0.5})
        async for msg in mqtt.messages():
            print(msg)
"""

import asyncio
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional

import paho.mqtt.client as paho

from schema_api import MQTTConfig

logger = logging.getLogger(__name__)


# -------------------------------------------------
# Message container
# -------------------------------------------------

@dataclass
class MQTTMessage:
    topic: str
    payload: dict | str
    qos: int = 0


# -------------------------------------------------
# Connection state
# -------------------------------------------------

@dataclass
class ConnectionState:
    connected: bool = False
    reconnect_count: int = 0
    last_error: Optional[str] = None


# -------------------------------------------------
# MQTT Async Client
# -------------------------------------------------

class MQTTAsyncClient:
    """
    Async-friendly MQTT client built on paho-mqtt.

    paho runs in its own thread; messages are forwarded
    to asyncio via a thread-safe Queue.
    """

    _RECONNECT_BASE   = 1.0   # seconds
    _RECONNECT_MAX    = 30.0  # seconds cap
    _QUEUE_MAX        = 256   # drop oldest when full

    def __init__(self, config: MQTTConfig):
        self._cfg    = config
        self._loop   = asyncio.get_event_loop()
        self._queue: asyncio.Queue[MQTTMessage] = asyncio.Queue(
            maxsize=self._QUEUE_MAX
        )
        self._state  = ConnectionState()
        self._stop   = threading.Event()

        # paho client
        self._client = paho.Client(
            client_id=config.client_id,
            clean_session=True,
        )
        if config.username:
            self._client.username_pw_set(config.username, config.password)

        self._client.on_connect    = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message    = self._on_message

    # -------------------------------------------------
    # Context manager
    # -------------------------------------------------

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, *_):
        await self.disconnect()

    # -------------------------------------------------
    # Connect / disconnect
    # -------------------------------------------------

    async def connect(self):
        """Connect to broker and start background loop."""
        await asyncio.get_event_loop().run_in_executor(
            None, self._connect_sync
        )
        logger.info(
            "MQTT connected  broker=%s:%s  client=%s",
            self._cfg.broker, self._cfg.port, self._cfg.client_id,
        )

    def _connect_sync(self):
        self._client.connect(
            self._cfg.broker,
            self._cfg.port,
            self._cfg.keepalive,
        )
        self._client.loop_start()

    async def disconnect(self):
        """Graceful shutdown."""
        self._stop.set()
        await asyncio.get_event_loop().run_in_executor(
            None, self._disconnect_sync
        )
        logger.info("MQTT disconnected")

    def _disconnect_sync(self):
        self._client.loop_stop()
        self._client.disconnect()

    # -------------------------------------------------
    # Publish helpers
    # -------------------------------------------------

    async def publish_control(self, payload: dict) -> None:
        """
        Publish a control action to the control topic.
        QoS 1 — at least once delivery.
        """
        await self._publish(
            topic   = self._cfg.topics.control,
            payload = payload,
            qos     = 1,
        )

    async def publish_emergency_stop(self) -> None:
        """
        Publish emergency stop.
        QoS 2 (exactly once) + retained so Jetson gets it even after reconnect.
        """
        await self._publish(
            topic    = self._cfg.topics.emergency,
            payload  = {"cmd": "STOP", "ts": time.time()},
            qos      = 2,
            retain   = True,
        )
        logger.warning("EMERGENCY STOP published")

    async def clear_emergency(self) -> None:
        """Clear retained emergency stop by publishing empty retained msg."""
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self._client.publish(
                self._cfg.topics.emergency, payload=None,
                qos=2, retain=True,
            ),
        )
        logger.info("Emergency stop cleared")

    async def _publish(
        self,
        topic: str,
        payload: dict,
        qos: int = 1,
        retain: bool = False,
    ) -> None:
        raw = json.dumps(payload)
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self._client.publish(topic, raw, qos=qos, retain=retain),
        )
        logger.debug("PUBLISH  topic=%s  qos=%d  payload=%s", topic, qos, raw)

    # -------------------------------------------------
    # Subscribe / message stream
    # -------------------------------------------------

    def _subscribe_all(self):
        """Subscribe to state + telemetry topics."""
        topics = [
            (self._cfg.topics.state,     self._cfg.qos),
            (self._cfg.topics.telemetry, self._cfg.qos),
            (self._cfg.topics.emergency, 2),
        ]
        for topic, qos in topics:
            self._client.subscribe(topic, qos)
            logger.info("Subscribed  topic=%s  qos=%d", topic, qos)

    async def messages(self) -> AsyncIterator[MQTTMessage]:
        """
        Async generator — yield incoming messages.

        Usage:
            async for msg in mqtt.messages():
                handle(msg)
        """
        while not self._stop.is_set():
            try:
                msg = await asyncio.wait_for(
                    self._queue.get(), timeout=1.0
                )
                yield msg
            except asyncio.TimeoutError:
                continue

    # -------------------------------------------------
    # paho callbacks  (called from paho thread)
    # -------------------------------------------------

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self._state.connected    = True
            self._state.reconnect_count = 0
            self._state.last_error   = None
            self._subscribe_all()
            logger.info("MQTT on_connect  rc=0  OK")
        else:
            self._state.last_error = f"connect failed rc={rc}"
            logger.error("MQTT on_connect  rc=%d", rc)

    def _on_disconnect(self, client, userdata, rc):
        self._state.connected = False
        if rc != 0 and not self._stop.is_set():
            logger.warning("MQTT unexpected disconnect rc=%d — reconnecting", rc)
            self._schedule_reconnect()

    def _on_message(self, client, userdata, message):
        try:
            payload = json.loads(message.payload.decode())
        except (json.JSONDecodeError, UnicodeDecodeError):
            payload = message.payload.decode(errors="replace")

        msg = MQTTMessage(
            topic   = message.topic,
            payload = payload,
            qos     = message.qos,
        )

        # thread-safe put into asyncio queue
        try:
            self._loop.call_soon_threadsafe(
                self._queue.put_nowait, msg
            )
        except asyncio.QueueFull:
            logger.warning("MQTT queue full — dropping message topic=%s", message.topic)

    # -------------------------------------------------
    # Auto-reconnect  (exponential back-off)
    # -------------------------------------------------

    def _schedule_reconnect(self):
        thread = threading.Thread(
            target=self._reconnect_loop, daemon=True
        )
        thread.start()

    def _reconnect_loop(self):
        delay = min(
            self._RECONNECT_BASE * (2 ** self._state.reconnect_count),
            self._RECONNECT_MAX,
        )
        self._state.reconnect_count += 1
        logger.info(
            "Reconnect attempt %d in %.1fs",
            self._state.reconnect_count, delay,
        )
        time.sleep(delay)
        if not self._stop.is_set():
            try:
                self._client.reconnect()
            except Exception as exc:
                logger.error("Reconnect failed: %s", exc)
                self._schedule_reconnect()

    # -------------------------------------------------
    # Properties
    # -------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self._state.connected

    @property
    def state(self) -> ConnectionState:
        return self._state