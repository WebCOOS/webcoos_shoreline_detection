from enum import Enum


class MethodFramework(str, Enum):
    skimage = "skimage"


class MethodName(str, Enum):
    shoreline_otsu = "shoreline_otsu"


class ShorelineOtsuVersion(str, Enum):
    v1 = "v1"
