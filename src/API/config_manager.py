#my_project\src\API\loder_create_config_api.py
from src.API.src_api.loader_api import ConfigLoader
from src.API.src_api.store_api import ConfigStore

class ConfigManager:

    def __init__(self, path):

        self.loader = ConfigLoader(path)
        self.store = ConfigStore()

        self.reload()

    def reload(self):

        config = self.loader.load()

        self.store.set(config)

        print("Config reloaded")

    def get(self):

        return self.store.get()

    def update(self, key, value):

        config = self.store.get()

        config[key] = value

        self.store.set(config)

# =====================================
# TEST
# =====================================

if __name__ == "__main__":

    print("=== Config Manager Test ===")

    manager = ConfigManager(r"D:\Project_end\New_world\my_project\src\API\config\agent.yaml")

    print("\nInitial Config:")
    print(manager.get())

    print("\nReload config...")
    manager.reload()

    print("\nConfig after reload:")
    print(manager.get())

    print("\nUpdate runtime value...")

    manager.update("debug_mode", True)

    print(manager.get())