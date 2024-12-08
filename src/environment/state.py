from enum import Enum


class NodeState(Enum):
    SAFE = "safe"
    COMPROMISED = "compromised"
    OVERLOADED = "overloaded"
    ISOLATED = "isolated"
    SCANNING = "scanning"