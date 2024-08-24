from ExpertRecSystem.system.collaboration import CollaborationSystem
from ExpertRecSystem.system.base import System

SYSTEMS: list[type[System]] = [
    value
    for value in globals().values()
    if isinstance(value, type) and issubclass(value, System) and value != System
]
