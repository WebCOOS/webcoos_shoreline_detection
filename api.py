import os
# import requests
from typing import Any
from pathlib import Path
from pydantic import BaseModel
from fastapi import FastAPI, Depends, UploadFile, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from shoreline_otsu_processing import shoreline_otsu_process_image, SKIMAGE_METHODS
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
    str(Path(__file__).with_name('outputs') / 'fastapi')
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
    f"{SKIMAGE_ENDPOINT_PREFIX}/{{method}}/{{version}}/upload",
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
):
    """Perform shoreline detection based on selected method and method version."""
    bytedata = file.file.read()

    ( name, ext ) = namify_for_content( bytedata )

    assert ext in ALLOWED_IMAGE_EXTENSIONS, \
        f"{ext} not in allowed image file types: {repr(ALLOWED_IMAGE_EXTENSIONS)}"

    ( plot_res_path, json_res_path, detection_result ) = shoreline_otsu_process_image(
        shoreline_method,
        output_path,
        method,
        version,
        shoreline_name,
        name,
        bytedata
    )

    if( res_path is None ):
        return annotation_image_and_detection_result(
            None,
            detection_result
        )

    rel_path = os.path.relpath( res_path, output_path )

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

    return annotation_image_and_detection_result(
        url_path_for_output,
        detection_result
    )


@app.post("/health")
def health():
    return { "health": "ok" }
