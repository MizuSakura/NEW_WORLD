# reward_types/__init__.py

from .continuous import ContinuousReward
from .hybrid import HybridReward
from .control_v1 import ControlV1Reward
from .pid_stable import PIDStableReward
from .zone_stable import ZoneStableReward
from .pid_adaptive import PIDAdaptiveReward
REWARD_REGISTRY = {
    "continuous": ContinuousReward,
    "hybrid": HybridReward,
    "control_v1": ControlV1Reward,
    "pid_stable": PIDStableReward,
    "zone_stable": ZoneStableReward,
    "pid_adiaptive" : PIDAdaptiveReward,
}