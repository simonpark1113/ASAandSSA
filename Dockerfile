FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

COPY requirements.txt .

RUN pip install --default-timeout=100 -r requirements.txt

WORKDIR /root

CMD jupyter lab --no-browser --allow-root