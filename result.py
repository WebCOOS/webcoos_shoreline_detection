
from dataclasses import dataclass

# @dataclass
# class BoundingBoxPoint:
#     x: float
#     y: float


@dataclass
class ShorelineDetectionResult():

    # Example: skimage, shoreline_otsu, v1
    detection_model_framework: str
    detection_model_name: str
    detection_model_version: str

    shoreline_name: str

    detected_shoreline: dict = None

    shoreline_plot_uri: str = None

    is_valid: bool = None
