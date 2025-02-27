# $ docker run --gpus all --ipc=host -it -d -v $(pwd):/app -p 10888:8888 \
#     --name spark_brasyn spark_brasyn:latest

docker run --gpus all --ipc=host -v $(pwd):/root -v /home/cgv/Spark/data:/data_folder -p 6007:6006 -p 10889:8888 -it -d  --name spark_nnunet spark_nnunet:latest /bin/bash