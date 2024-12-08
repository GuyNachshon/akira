from enum import Enum


class NetworkAction(Enum):
    SCAN = "scan"
    ISOLATE = "isolate"
    RESTORE = "restore"
    PATCH = "patch"
    MONITOR = "monitor"

    @classmethod
    def from_index(cls, index: int) -> 'NetworkAction':
        """Convert numeric action to NetworkAction enum."""
        try:
            return list(cls)[index]
        except IndexError:
            raise ValueError(f"{index} is not a valid NetworkAction")

    @classmethod
    def get_action_space_size(cls) -> int:
        """Get the size of the action space."""
        return len(cls)
