
from fastapi.testclient import TestClient
from api import app


test_http_client = TestClient( app )


class TestHealthEndpoint:

    def test_health(self):

        response = test_http_client.get("/health")

        assert response.status_code == 200, \
            "Must succeed with HTTP 200 response"

        assert response.json() == {"health":"ok"}, \
            "Must response with {'health':'ok'} message"
