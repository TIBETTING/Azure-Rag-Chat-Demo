FROM python:3.10-bullseye

WORKDIR /azure-chat

RUN apt-get update && apt-get -y upgrade

COPY . /azure-chat
EXPOSE 8501

RUN pip install -r ./requirements-dev.txt