# WebCOOS Shoreline Detection

API for performing shoreline detection on pre-configured shoreline imagery.

Developed by and collaborated with:

*   Jeremy Braun (UNCW) (jeb2694@uncw.edu)
*   Joseph Long (UNCW) (longjw@uncw.edu)
*   Emily Tango-Lee (UNCW) (ejt4639@uncw.edu)

![Shoreline Example](example.jpg "Shoreline Example")

## Local setup

It is recommended to create a new Python virtual environment using `conda`, or
its faster alternative, `micromamba`.

`conda` environment creation and activation:

```shell
conda env create -f environment.yml
conda activate webcoos_shoreline_detection
```

`micromamba` environment creation and activation:

```shell
micromamba create -f environment.yml
micromamba activate webcoos_shoreline_detection
```

## FastAPI Serving

The models can be served using a FastAPI server. The server allows the POSTing
of raw image (file uploads) to run against an available shoreline detection
method and set of known shoreline parameters.

The method used for shoreline detection to use is supplied as URL path
parameters, with the general endpoint scheme being the following:

```shell
 # Image file upload endpoint
POST /{method_framework}/{method_name}/{method_version}/{shoreline_name}/upload

# Example to upload an oak island image and to detect using the latest Otsu
# method for shoreline detection (which uses scikit as a based framework).
POST /skimage/shoreline_otsu/v1/oakisland_west/upload
```

The server can be started with either:

```shell
uvicorn api:app --port 8778
```

...or use `modd` (from [cortesi/modd](https://github.com/cortesi/modd))

```shell
modd
```

...or FastAPI server can also be served using Docker:

```shell
docker compose up
```

### OpenAPI Spec / UI (Swagger)

The FastAPI framework provides automatically generated
[OpenAPI](https://spec.openapis.org/oas/v3.1.0) specifications for the
endpoints and calling parameters for all endpoints within the service.

The spec and the generated user interface for that spec can be viewed with the
service running, accessible via <http://localhost:8778/docs>

## Testing

The FastAPI service can be tested with `pytest`, which will run the battery of unit and
integration tests within the codebase.

```shell
pytest
```
