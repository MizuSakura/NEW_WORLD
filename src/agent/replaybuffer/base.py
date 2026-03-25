#my_project\src\agent\replaybuffer\base.py
class BaseReplayBuffer:
    def push(self, *args):
        raise NotImplementedError

    def sample(self, batch_size):
        raise NotImplementedError

    def update_priorities(self, *args):
        pass  

    def __len__(self):
        raise NotImplementedError