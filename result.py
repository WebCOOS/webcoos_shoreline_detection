
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import numpy as np


@dataclass
class BoundingBoxPoint:
    x: float
    y: float


@dataclass
class ShorelineDetectionResult():

    # Example: skimage, shoreline_otsu, v1
    detection_model_framework: str
    detection_model_name: str
    detection_model_version: str

    shoreline_name: str

    # def add(
    #     self,
    #     classification_name: str,
    #     classification_score: float,
    #     bbox: Tuple[BoundingBoxPoint, BoundingBoxPoint] = None
    # ):
    #     assert classification_name and isinstance( classification_name, str )
    #     assert isinstance( classification_score, ( float, np.float32, np.float64 ) ), \
    #         f"classification score should be float, got {classification_score.__class__.__name__}"

    #     classification_score = float( classification_score )

    #     if bbox is not None:
    #         assert isinstance( bbox, tuple )
    #         assert all( [ isinstance( b, BoundingBoxPoint ) for b in bbox ] )

    #     self.classification_scores.append(
    #         {
    #             classification_name: classification_score
    #         }
    #     )

    #     if bbox is not None:

    #         self.classification_bboxes.append( bbox )

    #     self.detection_count += 1

    #     if( self.detection_count > 0 ):
    #         self.detected = True

    #     return
