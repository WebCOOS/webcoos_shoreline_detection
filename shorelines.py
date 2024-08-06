from enum import Enum


class Shoreline(str, Enum):
    """Pre-determined set of shoreline descriptors that map to the available"""
    """JSON configuration files."""

    oakisland_west = "oakisland_west"
    currituck_hampton_inn = "currituck_hampton_inn"
