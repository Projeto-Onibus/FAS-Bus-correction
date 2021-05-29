FROM nvidia/cuda:11.2.0-cudnn8-runtime-ubuntu20.04

RUN apt update && apt install -y python3 python3-pip libpq-dev

RUN python3 -m pip install numpy pandas cupy-cuda112 psycopg2

COPY ./app/* /app/
