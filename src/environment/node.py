from dataclasses import dataclass
from .state import NodeState


@dataclass
class Node:
    id: str
    vulnerability: float  # 0-10
    state: NodeState
    value: float  # Business value of the node
    last_scan_time: int
    isolation_time: int

    def is_high_value(self) -> bool:
        return self.vulnerability > 7 or self.value > 7