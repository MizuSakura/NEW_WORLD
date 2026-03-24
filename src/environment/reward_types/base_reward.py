# reward_types/base_reward.py

class BaseReward:
    """
    Base class for all reward functions

    ทุก reward จะ access ข้อมูลผ่าน manager (self.m)
    """

    def __init__(self, manager):
        self.m = manager

    def compute(self):
        raise NotImplementedError("Reward must implement compute()")