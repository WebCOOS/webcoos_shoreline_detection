
from fastapi.testclient import TestClient
from api import app
from method_version import MethodFramework, MethodName, ShorelineOtsuVersion
from shorelines import Shoreline
import pathlib

test_http_client = TestClient( app )

OAKISLAND_WEST_TEST_INPUT_FILE="timex.oakisland_west-2024-06-25-180320Z.jpg"
oakisland_west_test_input_file_path = (
    pathlib.Path( __file__ ).parent
        / '..'
        / 'inputs'
        / OAKISLAND_WEST_TEST_INPUT_FILE
)

class TestShorelineOtsuEndpoints:

    def test_upload(self):

        method_framework = MethodFramework.skimage.value
        method_name = MethodName.shoreline_otsu.value
        method_version = ShorelineOtsuVersion.v1.value
        oakisland_west = Shoreline.oakisland_west.value

        endpoint_path = f"/{method_framework}/{method_name}/{method_version}/{oakisland_west}/upload"

        assert endpoint_path == '/skimage/shoreline_otsu/v1/oakisland_west/upload'

        assert oakisland_west_test_input_file_path.exists()
        assert oakisland_west_test_input_file_path.is_file()

        with open( str( oakisland_west_test_input_file_path ), 'rb' ) as fh:
            response = test_http_client.post(
                endpoint_path,
                headers = {
                    # 'Content-Type': 'multipart/form-data',
                    'Accept': 'application/json'
                },
                files = {
                    'file': fh
                }
            )

        assert response.status_code == 200, \
            (
                "Must succeed with HTTP 200 response, got "
                f"{response.status_code}, response: {response.text}"
            )

        response_data = response.json()

        DETECTED_SHORELINE='detected_shoreline'
        DETECTION_MODEL_NAME='detection_model_name'
        SHORELINE_POINTS='Shoreline Points'
        SHORELINE_PLOT_URI='shoreline_plot_uri'

        assert DETECTION_MODEL_NAME in response_data
        assert response_data[DETECTION_MODEL_NAME] == 'shoreline_otsu'

        assert DETECTED_SHORELINE in response_data
        assert SHORELINE_POINTS in response_data[DETECTED_SHORELINE]

        assert len( response_data[DETECTED_SHORELINE][SHORELINE_POINTS] ) > 0

        assert SHORELINE_PLOT_URI in response_data
        assert str( response_data[SHORELINE_PLOT_URI] ).startswith( 'http' )

    def test_invalid_upload(self):

        method_framework = MethodFramework.skimage.value
        method_name = MethodName.shoreline_otsu.value
        method_version = ShorelineOtsuVersion.v1.value
        oakisland_west = Shoreline.oakisland_west.value

        endpoint_path = f"/{method_framework}/{method_name}/{method_version}/{oakisland_west}/upload"

        assert endpoint_path == '/skimage/shoreline_otsu/v1/oakisland_west/upload'

        assert oakisland_west_test_input_file_path.exists()
        assert oakisland_west_test_input_file_path.is_file()

        with open( str( oakisland_west_test_input_file_path ), 'rb' ) as fh:
            response = test_http_client.post(
                endpoint_path,
                headers = {
                    # 'Content-Type': 'multipart/form-data',
                    'Accept': 'application/json'
                },
                params={
                    # Request a very high minimum number of points to guarantee
                    # an invalid result.
                    'minimum_shoreline_points': 100
                },
                files = {
                    'file': fh
                }
            )

        assert response.status_code == 200, \
            (
                "Must succeed with HTTP 200 response, got "
                f"{response.status_code}, response: {response.text}"
            )

        response_data = response.json()

        IS_VALID='is_valid'

        assert IS_VALID in response_data
        assert response_data[IS_VALID] == False
