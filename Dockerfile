FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
MAINTAINER simonp6@kaist.ac.kr

COPY requirements.txt .

RUN pip install --default-timeout=100 -r requirements.txt

# COPY nnUNet /nnUNet
# RUN cd /nnUNet && pip install --default-timeout=1000 -e .


# workdir을 /app으로 설정
WORKDIR /root

CMD jupyter lab --no-browser --allow-root