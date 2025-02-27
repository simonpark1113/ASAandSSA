CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 2 3d_fullres 0 --npz & # train on GPU 0
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 1 3d_fullres 1 --npz & # train on GPU 1
CUDA_VISIBLE_DEVICES=2 nnUNetv2_train 1 3d_fullres 2 --npz & # train on GPU 2
CUDA_VISIBLE_DEVICES=3 nnUNetv2_train 1 3d_fullres 3 --npz & # train on GPU 3
wait