
from prometheus_client import (
    make_asgi_app,
    CollectorRegistry,
    multiprocess,
    Counter
)
from method_version import (
    MethodFramework,
    MethodName,
    ShorelineOtsuVersion,
)
from shorelines import Shoreline
import os


SHORELINE_DETECTION_COUNTER = Counter(
    'shoreline_detection_counter',
    'Overall count of inputs against a specific shoreline',
    [
        'method_framework',
        'method_name',
        'method_version',
        'shoreline_name',
    ]
)

# Per: <https://prometheus.github.io/client_python/instrumenting/labels/>
#   Metrics with labels are not initialized when declared, because the client
#   can’t know what values the label can have. It is recommended to initialize
#   the label values by calling the .labels() method alone:
#
#       c.labels('get', '/')

SHORELINES = [
    (
        MethodFramework.skimage,
        MethodName.shoreline_otsu,
        ShorelineOtsuVersion.v1,
        sl.value
    ) for sl in Shoreline
]

for ( fw, mdl, ver, sl_name ) in SHORELINES:
    # Initialize counters
    SHORELINE_DETECTION_COUNTER.labels(
        fw.name,
        mdl.value,
        ver.value,
        sl_name,
    )


PROMETHEUS_MULTIPROC_DIR = 'PROMETHEUS_MULTIPROC_DIR'


def make_metrics_app():
    registry = CollectorRegistry()

    # Try to detect and provide default, suitable PROMETHEUS_MULTIPROC_DIR
    # even if the environment variable isn't set.
    pmd = os.getenv( PROMETHEUS_MULTIPROC_DIR, None )

    if pmd is None:
        pmd = '/tmp'
        os.environ[ PROMETHEUS_MULTIPROC_DIR ] = pmd

    multiprocess.MultiProcessCollector( registry )
    return make_asgi_app( registry = registry )


def increment_shoreline_counter(
    fw: str,
    mth_name: str,
    mth_version: str,
    shoreline_name: str
):
    SHORELINE_DETECTION_COUNTER.labels(
        fw,
        mth_name,
        mth_version,
        shoreline_name
    ).inc()
