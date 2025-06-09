
from fastapi.testclient import TestClient
from api import app
from method_version import MethodFramework, MethodName, ShorelineOtsuVersion
from shorelines import Shoreline
import pathlib

test_http_client = TestClient( app )

TEST_DIR=pathlib.Path( __file__ ).parent

OAKISLAND_WEST_TEST_INPUT_FILE="timex.oakisland_west-2024-06-25-180320Z.jpg"
oakisland_west_test_input_file_path = (
    TEST_DIR
        / '..'
        / 'inputs'
        / OAKISLAND_WEST_TEST_INPUT_FILE
)

CURRITUCK_HAMPTON_INN_TEST_INPUT_FILE="timex.currituck_hampton_inn-2024-07-12-001729Z.jpg"
currituck_hampton_inn_test_input_file_path = (
    TEST_DIR
        / '..'
        / 'inputs'
        / CURRITUCK_HAMPTON_INN_TEST_INPUT_FILE
)

ADDITIONAL_TEST_BATTERY=(
    ( 'currituck_sailfish', TEST_DIR / '..' / 'inputs' / 'timex.currituck_sailfish-2025-06-06-182008Z.jpg' ),
    ( 'jennette_north', TEST_DIR / '..' / 'inputs' / 'timex.jennette_north-2025-05-18-172859Z.jpg' ),
    ( 'jennette_south', TEST_DIR / '..' / 'inputs' / 'timex.jennette_south-2025-05-16-200448Z.jpg' ),
)


class TestShorelineOtsuEndpoints:

    def test_upload_oakisland_west(self):

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

    def test_upload_currituck_hampton_inn(self):

        method_framework = MethodFramework.skimage.value
        method_name = MethodName.shoreline_otsu.value
        method_version = ShorelineOtsuVersion.v1.value
        currituck_hampton_inn = Shoreline.currituck_hampton_inn.value

        endpoint_path = f"/{method_framework}/{method_name}/{method_version}/{currituck_hampton_inn}/upload"

        assert endpoint_path == '/skimage/shoreline_otsu/v1/currituck_hampton_inn/upload'

        assert currituck_hampton_inn_test_input_file_path.exists()
        assert currituck_hampton_inn_test_input_file_path.is_file()

        with open( str( currituck_hampton_inn_test_input_file_path ), 'rb' ) as fh:
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


    def test_upload_additional_test_battery(self):

        method_framework = MethodFramework.skimage.value
        method_name = MethodName.shoreline_otsu.value
        method_version = ShorelineOtsuVersion.v1.value

        for ( station_name_str, input_path ) in ADDITIONAL_TEST_BATTERY:

            station_name = Shoreline( station_name_str )

            endpoint_path = f"/{method_framework}/{method_name}/{method_version}/{station_name}/upload"

            assert endpoint_path == f'/skimage/shoreline_otsu/v1/{station_name}/upload'

            assert input_path.exists()
            assert input_path.is_file()

            with open( str( input_path ), 'rb' ) as fh:
                response = test_http_client.post(
                    endpoint_path,
                    headers = {
                        # 'Content-Type': 'multipart/form-data',
                        'Accept': 'application/json'
                    },
                    files = {
                        'file': fh
                    },
                    params={
                        'minimum_shoreline_points': 30
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
            IS_VALID='is_valid'

            assert DETECTION_MODEL_NAME in response_data
            assert response_data[DETECTION_MODEL_NAME] == 'shoreline_otsu'

            assert DETECTED_SHORELINE in response_data
            assert SHORELINE_POINTS in response_data[DETECTED_SHORELINE]

            assert len( response_data[DETECTED_SHORELINE][SHORELINE_POINTS] ) > 0

            assert SHORELINE_PLOT_URI in response_data
            assert str( response_data[SHORELINE_PLOT_URI] ).startswith( 'http' )

            assert IS_VALID in response_data
            assert response_data[IS_VALID] is True

    def test_invalid_upload_oakisland_west(self):

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


    # def test_upload_mismatched_shoreline(self):

    #     # Upload a Oak Island picture to the Currituck Hampton Inn config,
    #     # ensure that behavior is as expected.

    #     method_framework = MethodFramework.skimage.value
    #     method_name = MethodName.shoreline_otsu.value
    #     method_version = ShorelineOtsuVersion.v1.value
    #     currituck_hampton_inn = Shoreline.currituck_hampton_inn.value

    #     endpoint_path = f"/{method_framework}/{method_name}/{method_version}/{currituck_hampton_inn}/upload"

    #     assert endpoint_path == '/skimage/shoreline_otsu/v1/currituck_hampton_inn/upload'

    #     # Here, we submit the *wrong* file
    #     assert oakisland_west_test_input_file_path.exists()
    #     assert oakisland_west_test_input_file_path.is_file()

    #     with open( str( oakisland_west_test_input_file_path ), 'rb' ) as fh:
    #         response = test_http_client.post(
    #             endpoint_path,
    #             headers = {
    #                 # 'Content-Type': 'multipart/form-data',
    #                 'Accept': 'application/json'
    #             },
    #             files = {
    #                 'file': fh
    #             }
    #         )

    #     assert response.status_code == 200, \
    #         (
    #             "Must succeed with HTTP 200 response, got "
    #             f"{response.status_code}, response: {response.text}"
    #         )

    #     response_data = response.json()

    #     DETECTED_SHORELINE='detected_shoreline'
    #     DETECTION_MODEL_NAME='detection_model_name'
    #     SHORELINE_POINTS='Shoreline Points'
    #     SHORELINE_PLOT_URI='shoreline_plot_uri'

    #     assert DETECTION_MODEL_NAME in response_data
    #     assert response_data[DETECTION_MODEL_NAME] == 'shoreline_otsu'

    #     assert DETECTED_SHORELINE in response_data
    #     assert SHORELINE_POINTS in response_data[DETECTED_SHORELINE]

    #     assert len( response_data[DETECTED_SHORELINE][SHORELINE_POINTS] ) > 0

    #     assert SHORELINE_PLOT_URI in response_data
    #     assert str( response_data[SHORELINE_PLOT_URI] ).startswith( 'http' )
