#my_project\src\API\src_api\store_api.py
import threading

class ConfigStore:

    def __init__(self):

        self._config = {}
        self._lock = threading.Lock()

    def set(self, config):

        with self._lock:
            self._config = config

    def get(self):

        with self._lock:
            return self._config

    def update(self, key, value):

        with self._lock:
            self._config[key] = value

# =====================================
# TEST
# =====================================

if __name__ == "__main__":

    print("=== Store Test ===")

    store = ConfigStore()

    test_config = {
        "mqtt": {
            "broker": "localhost",
            "port": 1883
        }
    }

    print("Setting config...")
    store.set(test_config)

    print("Current config:")
    print(store.get())

    print("\nUpdating broker...")

    store.update("mqtt", {"broker": "192.168.1.100", "port": 1883})

    print("Updated config:")
    print(store.get())