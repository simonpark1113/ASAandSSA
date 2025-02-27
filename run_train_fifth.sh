CUDA_VISIBLE_DEVICES=0,1,2,3 nnUNetv2_train 13 3d_customSymOnlyProject 4 --npz -num_gpus 4 & # train on GPU 0
wait