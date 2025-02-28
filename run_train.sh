CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 5 3d_bothATTMS_noOS 0 --npz & # train on GPU 0
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 5 3d_bothATTMS_noOS 1 --npz & # train on GPU 1
CUDA_VISIBLE_DEVICES=2 nnUNetv2_train 5 3d_bothATTMS_noOS 2 --npz & # train on GPU 2
CUDA_VISIBLE_DEVICES=3 nnUNetv2_train 5 3d_bothATTMS_noOS 3 --npz & # train on GPU 3
wait
CUDA_VISIBLE_DEVICES=4 nnUNetv2_train 5 3d_bothATTMS_noOS 4 --npz & # train on GPU 0
wait
