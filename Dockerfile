# syntax=docker/dockerfile:1

FROM --platform=linux/amd64 pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel
WORKDIR /root/code
COPY . .
RUN pip install --no-cache-dir --upgrade \
    pyg_lib \
    torch_scatter \
    torch_sparse \
    torch_cluster \
    torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.0.1+cu118.html
RUN pip install --no-cache-dir -r requirements.txt