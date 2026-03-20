"""
HTTP Async Client
-----------------

Async HTTP client for communicating with Nvidia Jetson.

Features
--------
- Upload model file  (.pt / .pth)
- Upload config file (.yaml)
- Download log / result file
- Health check endpoint

Built on aiohttp for non-blocking I/O.

Usage
-----
    async with HTTPAsyncClient(config) as http:
        ok = await http.health_check()
        await http.upload_model(Path("models/sac.pt"))
        await http.download_log("run_001.pt", dest=Path("logs/"))
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Optional

import aiohttp
from aiohttp import FormData

from schema_api import HTTPConfig

logger = logging.getLogger(__name__)


# -------------------------------------------------
# Upload result container
# -------------------------------------------------

class UploadResult:
    def __init__(self, success: bool, filename: str, message: str = "", elapsed: float = 0.0):
        self.success  = success
        self.filename = filename
        self.message  = message
        self.elapsed  = elapsed

    def __repr__(self):
        status = "OK" if self.success else "FAIL"
        return f"UploadResult({status}  file={self.filename}  {self.elapsed:.2f}s)"


# -------------------------------------------------
# HTTP Async Client
# -------------------------------------------------

class HTTPAsyncClient:
    """
    Async HTTP client — upload / download / health check.

    All methods are coroutines and safe to await concurrently.
    """

    _TIMEOUT_UPLOAD   = aiohttp.ClientTimeout(total=120)  # large model files
    _TIMEOUT_DEFAULT  = aiohttp.ClientTimeout(total=15)
    _CHUNK_SIZE       = 64 * 1024  # 64 KB read chunks

    _ALLOWED_MODEL_SUFFIXES = {".pt", ".pth"}
    _ALLOWED_CONFIG_SUFFIXES = {".yaml", ".yml"}

    def __init__(self, config: HTTPConfig):
        self._cfg     = config
        self._base    = f"http://{config.host}:{config.port}"
        self._session: Optional[aiohttp.ClientSession] = None

    # -------------------------------------------------
    # Context manager
    # -------------------------------------------------

    async def __aenter__(self):
        self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, *_):
        if self._session:
            await self._session.close()

    # -------------------------------------------------
    # Internal helpers
    # -------------------------------------------------

    def _url(self, path: str) -> str:
        return f"{self._base}{path}"

    def _assert_session(self):
        if self._session is None:
            raise RuntimeError(
                "HTTPAsyncClient must be used as async context manager"
            )

    def _validate_suffix(self, path: Path, allowed: set[str]):
        if path.suffix.lower() not in allowed:
            raise ValueError(
                f"File type '{path.suffix}' not allowed. "
                f"Expected: {allowed}"
            )

    # -------------------------------------------------
    # Health check
    # -------------------------------------------------

    async def health_check(self) -> bool:
        """
        GET /health — returns True if Jetson server is reachable and OK.
        """
        self._assert_session()
        try:
            async with self._session.get(
                self._url("/health"),
                timeout=self._TIMEOUT_DEFAULT,
            ) as resp:
                ok = resp.status == 200
                logger.info("Health check  status=%d  ok=%s", resp.status, ok)
                return ok
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            logger.warning("Health check failed: %s", exc)
            return False

    # -------------------------------------------------
    # Upload model
    # -------------------------------------------------

    async def upload_model(self, path: Path) -> UploadResult:
        """
        POST /upload/model — stream .pt / .pth to Jetson.

        Streams the file in chunks so large models don't
        load fully into memory.
        """
        self._assert_session()
        self._validate_suffix(path, self._ALLOWED_MODEL_SUFFIXES)

        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        logger.info("Uploading model  file=%s  size=%d bytes", path.name, path.stat().st_size)
        t0 = time.monotonic()

        data = FormData()
        data.add_field(
            "file",
            open(path, "rb"),
            filename=path.name,
            content_type="application/octet-stream",
        )

        try:
            async with self._session.post(
                self._url("/3/model"),
                data=data,
                timeout=self._TIMEOUT_UPLOAD,
            ) as resp:
                elapsed = time.monotonic() - t0
                body    = await resp.json()
                success = resp.status == 200
                result  = UploadResult(
                    success  = success,
                    filename = path.name,
                    message  = body.get("message", ""),
                    elapsed  = elapsed,
                )
                if success:
                    logger.info("Model uploaded  %s", result)
                else:
                    logger.error("Model upload failed  status=%d  %s", resp.status, body)
                return result

        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            elapsed = time.monotonic() - t0
            logger.error("Model upload error: %s", exc)
            return UploadResult(
                success=False, filename=path.name,
                message=str(exc), elapsed=elapsed,
            )

    # -------------------------------------------------
    # Upload config
    # -------------------------------------------------

    async def upload_config(self, path: Path) -> UploadResult:
        """
        POST /upload/config — send .yaml config to Jetson.
        """
        self._assert_session()
        self._validate_suffix(path, self._ALLOWED_CONFIG_SUFFIXES)

        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        logger.info("Uploading config  file=%s", path.name)
        t0 = time.monotonic()

        data = FormData()
        data.add_field(
            "file",
            open(path, "rb"),
            filename=path.name,
            content_type="application/x-yaml",
        )

        try:
            async with self._session.post(
                self._url("/upload/config"),
                data=data,
                timeout=self._TIMEOUT_DEFAULT,
            ) as resp:
                elapsed = time.monotonic() - t0
                body    = await resp.json()
                success = resp.status == 200
                result  = UploadResult(
                    success=success, filename=path.name,
                    message=body.get("message", ""), elapsed=elapsed,
                )
                logger.info("Config upload  %s", result)
                return result

        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            elapsed = time.monotonic() - t0
            logger.error("Config upload error: %s", exc)
            return UploadResult(
                success=False, filename=path.name,
                message=str(exc), elapsed=elapsed,
            )

    # -------------------------------------------------
    # Download log / result
    # -------------------------------------------------

    async def download_log(
        self,
        remote_filename: str,
        dest: Path,
    ) -> Path:
        """
        GET /download/{filename} — save file to local dest directory.

        Returns the path of the saved file.
        """
        self._assert_session()
        dest.mkdir(parents=True, exist_ok=True)

        url       = self._url(f"/download/{remote_filename}")
        save_path = dest / remote_filename

        logger.info("Downloading  remote=%s  dest=%s", remote_filename, save_path)
        t0 = time.monotonic()

        try:
            async with self._session.get(
                url, timeout=self._TIMEOUT_UPLOAD
            ) as resp:
                if resp.status != 200:
                    raise RuntimeError(
                        f"Download failed  status={resp.status}  file={remote_filename}"
                    )
                with open(save_path, "wb") as f:
                    async for chunk in resp.content.iter_chunked(self._CHUNK_SIZE):
                        f.write(chunk)

            elapsed = time.monotonic() - t0
            logger.info(
                "Download complete  file=%s  size=%d bytes  %.2fs",
                save_path, save_path.stat().st_size, elapsed,
            )
            return save_path

        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            logger.error("Download error: %s", exc)
            raise

    # -------------------------------------------------
    # Concurrent upload helpers
    # -------------------------------------------------

    async def upload_model_and_config(
        self,
        model_path: Path,
        config_path: Path,
    ) -> tuple[UploadResult, UploadResult]:
        """
        Upload model and config simultaneously.
        Returns (model_result, config_result).
        """
        model_result, config_result = await asyncio.gather(
            self.upload_model(model_path),
            self.upload_config(config_path),
        )
        return model_result, config_result
    
# -------------------------------------------------
# Main (test client)
# -------------------------------------------------

if __name__ == "__main__":

    import asyncio
    from pathlib import Path

    # ---------------------------------------------
    # Example config
    # ---------------------------------------------

    http_config = HTTPConfig(
        enable=True,
        host="127.0.0.1",
        port=8001
    )

    async def main():

        print("Starting HTTP async client test...\n")

        async with HTTPAsyncClient(http_config) as http:

            # -----------------------------
            # Health check
            # -----------------------------

            ok = await http.health_check()

            print("Health check:", ok)

            # -----------------------------
            # Upload model
            # -----------------------------

            model_path = Path("example_model.pt")

            if model_path.exists():

                result = await http.upload_model(model_path)

                print("Model upload:", result)

            else:

                print("Skip model upload (file not found)")

            # -----------------------------
            # Upload config
            # -----------------------------

            config_path = Path("example_config.yaml")

            if config_path.exists():

                result = await http.upload_config(config_path)

                print("Config upload:", result)

            else:

                print("Skip config upload (file not found)")

            # -----------------------------
            # Download log
            # -----------------------------

            try:

                path = await http.download_log(
                    "example_log.txt",
                    dest=Path("downloads")
                )

                print("Downloaded file:", path)

            except Exception as e:

                print("Download skipped:", e)


    asyncio.run(main())