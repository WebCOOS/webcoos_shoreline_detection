FROM mambaorg/micromamba:1.5.8-jammy
LABEL MAINTAINER="Josh Rhoades <josh@axds.co>"

ENV OUTPUT_DIRECTORY /outputs
ENV APP_DIRECTORY /app

ENV PROMETHEUS_MULTIPROC_DIR /tmp/metrics

USER root

#ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
        libgl1 \
        libglib2.0-0 \
        libegl-dev \
        libopengl0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
    mkdir ${APP_DIRECTORY} && \
    chown mambauser:mambauser ${APP_DIRECTORY} && \
    mkdir ${OUTPUT_DIRECTORY} && \
    chown mambauser:mambauser ${OUTPUT_DIRECTORY} && \
    mkdir ${PROMETHEUS_MULTIPROC_DIR} && \
    chown mambauser:mambauser ${PROMETHEUS_MULTIPROC_DIR}

#RUN pip install opencv-python

COPY --chown=mambauser:mambauser environment.yml /tmp/environment.yml
RUN --mount=type=cache,id=webcoos_shoreline_detector,target=/opt/conda/pkgs \
    --mount=type=cache,id=webcoos_shoreline_detector,target=/root/.cache/pip \
    micromamba install -c conda-forge --name base --yes --file /tmp/environment.yml && \
    micromamba clean --all --yes

ARG MAMBA_DOCKERFILE_ACTIVATE=1
ENV PATH "$MAMBA_ROOT_PREFIX/bin:$PATH"

# Copy Python app files
COPY --chown=mambauser:mambauser *.py /app/
# Embed the known station/shoreline file names within the image
RUN mkdir -p /app/cfg && chown mambauser:mambauser /app/cfg
COPY --chown=mambauser:mambauser cfg/*.json /app/cfg/

# Copy container-specific configuration
COPY --chown=mambauser:mambauser docker /docker/
RUN chmod u+x /docker/scripts/expire-annotated-images.sh

WORKDIR /app
CMD ["gunicorn", "api:app", "--config", "/docker/api/gunicorn.conf.py"]
