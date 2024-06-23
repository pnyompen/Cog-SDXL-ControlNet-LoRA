FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9
ENV PYTHONDONTWRITEBYTECODE 1
ARG MODEL_NAME
ARG REPLICATE_API_TOKEN

RUN apt-get update \
    && apt-get install -y curl
RUN curl -o /usr/local/bin/cog -L https://github.com/replicate/cog/releases/latest/download/cog_`uname -s`_`uname -m`
RUN chmod +x /usr/local/bin/cog

ENV DOCKER_CLIENT_VERSION=1.12.3
ENV DOCKER_API_VERSION=1.41
RUN curl -fsSL https://get.docker.com/ | sh

COPY . .

RUN pip install -r requirements.txt
RUN python script/download_weights.py

RUN echo $REPLICATE_API_TOKEN | cog login --token-stdin
RUN cog build
RUN cog push r8.im/${MODEL_NAME}

