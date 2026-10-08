from enum import Enum

try:
    StrEnum = __import__("enum").StrEnum
except AttributeError:
    class StrEnum(str, Enum):
        pass


class HarnessState(StrEnum):
    NEW = "NEW"
    INITIALIZED = "INITIALIZED"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    PAUSED_FOR_HUMAN = "PAUSED_FOR_HUMAN"
    VALIDATING = "VALIDATING"
    TARGET_REACHED = "TARGET_REACHED"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    SAFETY_STOP = "SAFETY_STOP"
    STOPPED_BY_USER = "STOPPED_BY_USER"
    FATAL_ERROR = "FATAL_ERROR"


TERMINAL_STATES = {
    HarnessState.TARGET_REACHED,
    HarnessState.BUDGET_EXHAUSTED,
    HarnessState.SAFETY_STOP,
    HarnessState.STOPPED_BY_USER,
    HarnessState.FATAL_ERROR,
}


# Experiment statuses that mean "work in flight"; anything else is settled.
ACTIVE_STATUSES = ("planned", "implemented", "running", "repairing")
