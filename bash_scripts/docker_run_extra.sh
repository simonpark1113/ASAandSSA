# $ docker run --gpus all --ipc=host -it -d -v $(pwd):/app -p 10888:8888 \
#     --name spark_brasyn spark_brasyn:latest

docker run --gpus all --ipc=host -v $(pwd):/root -v /home/cgv/Spark/data:/data_folder -it -d  --name spark_nnunet_extra spark_nnunet_extra:latest /bin/bash