from enum import auto, StrEnum


class Shoreline(StrEnum):
    """Pre-determined set of shoreline descriptors that map to the available"""
    """JSON configuration files."""

    oakisland_west = auto()
    currituck_hampton_inn = auto()
    currituck_sailfish = auto()
    ferrybeach_north = auto()
    jennette_north = auto()
    jennette_south = auto()
    westerly = auto()
