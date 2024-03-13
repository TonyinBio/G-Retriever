# syntax=docker/dockerfile:1

FROM --platform=linux/amd64 pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel
WORKDIR /root/code
COPY . .
RUN pip install --no-cache-dir --upgrade -r requirements.txt