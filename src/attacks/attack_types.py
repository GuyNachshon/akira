from enum import Enum


class AttackType(Enum):
    RANSOMWARE = "ransomware"
    APT = "apt"
    DDOS = "ddos"
    ZERODAY = "zeroday"
    LLM_DRIVEN = "llm_driven"