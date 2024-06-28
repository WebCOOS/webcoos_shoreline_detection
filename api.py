import os
# import requests
from typing import Any, Annotated
from pathlib import Path
from pydantic import BaseModel
from fastapi import FastAPI, Depends, UploadFile, Request, Query
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from shoreline_otsu_processing import (
    shoreline_otsu_process_image,
    SKIMAGE_METHODS,
    DEFAULT_MINIMUM_SHORELINE_DETECTION_POINTS
)
from metrics import make_metrics_app
from namify import namify_for_content
from result import ShorelineDetectionResult
from method_version import (
    MethodName,
    ShorelineOtsuVersion
)
from shorelines import Shoreline
import logging
from datetime import datetime, timezone

logger = logging.getLogger( __name__ )


app: FastAPI = FastAPI()
# Prometheus metrics
metrics_app = make_metrics_app()
app.mount("/metrics", metrics_app)


class UrlParams(BaseModel):
    url: str


SKIMAGE_ENDPOINT_PREFIX = "/skimage"
ALLOWED_IMAGE_EXTENSIONS = (
    "jpg",
    "png"
)

output_path = Path(os.environ.get(
    "OUTPUT_DIRECTORY",
    str(Path(__file__).with_name('outputs'))
))


def get_shoreline_method(method: MethodName, version: ShorelineOtsuVersion):
    return SKIMAGE_METHODS[method.value][version.value]


# Mounting the 'static' output files for the app
app.mount(
    "/outputs",
    StaticFiles(directory=output_path),
    name="outputs"
)


def annotation_image_and_detection_result(
    url: str,
    detection_result: ShorelineDetectionResult
):

    dt = datetime.now().replace( tzinfo=timezone.utc )
    dt_str = dt.isoformat( "T", "seconds" ).replace( '+00:00', 'Z' )

    return {
        "time": dt_str,
        "annotated_image_url": url,
        "detection_result": detection_result
    }


@app.get("/", include_in_schema=False)
async def index():
    """Convenience redirect to OpenAPI spec UI for service."""
    return RedirectResponse("/docs")


# YOLO object detection endpoints
@app.post(
    f"{SKIMAGE_ENDPOINT_PREFIX}/{{method}}/{{version}}/{{shoreline_name}}/upload",
    tags=['skimage'],
    summary="Shoreline Otsu shoreline prediction method against image upload",
)
def shoreline_otsu_from_upload(
    request: Request,
    method: MethodName,
    version: ShorelineOtsuVersion,
    shoreline_name: Shoreline,
    file: UploadFile,
    shoreline_method: Any = Depends(get_shoreline_method),
    minimum_shoreline_points: int|None = Annotated[
        int|None,
        Query(
            alias="minimum_shoreline_points",
            gt=0
        )
    ]
) -> ShorelineDetectionResult:
    """Perform shoreline detection based on selected method and method version."""
    bytedata = file.file.read()

    ( name, ext ) = namify_for_content( bytedata )

    assert ext in ALLOWED_IMAGE_EXTENSIONS, \
        f"{ext} not in allowed image file types: {repr(ALLOWED_IMAGE_EXTENSIONS)}"

    if (
        minimum_shoreline_points is None or
        not isinstance( minimum_shoreline_points, int )
    ):
        minimum_shoreline_points = DEFAULT_MINIMUM_SHORELINE_DETECTION_POINTS

    assert minimum_shoreline_points > 0, "minimum_shoreline_points must be > 0"

    detection_result: ShorelineDetectionResult = shoreline_otsu_process_image(
        shoreline_method,
        output_path,
        method,
        version,
        shoreline_name,
        name,
        bytedata,
        minimum_shoreline_points
    )

    if detection_result.shoreline_plot_uri is not None:
        # Attempt to give the shoreline path as helpful of a URI/URL as
        # possible, relying on the FastAPI routing to be able to provide a URL
        # to the file in the event that it is mounted under the 'outputs'
        # directory
        rel_path = os.path.relpath(
            detection_result.shoreline_plot_uri,
            output_path
        )

        url_path_for_output = rel_path

        try:
            # Try for an absolute URL (prefixed with http(s)://hostname, etc.)
            url_path_for_output = str( request.url_for( 'outputs', path=rel_path ) )
        except Exception:
            # Fall back to the relative URL determined by the router
            url_path_for_output = app.url_path_for(
                'outputs', path=rel_path
            )
        finally:
            pass

        detection_result.shoreline_plot_uri = url_path_for_output

    return detection_result


@app.get("/health")
def health():
    return { "health": "ok" }
