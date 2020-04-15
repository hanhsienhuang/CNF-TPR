python train_cnf.py --data mnist   --dims 64,64,64 --strides 1,1,1,1 --num_blocks 2 --layer_type concat --multiscale True --rademacher True --save experiments/mnist --adjoint --nonlinearity tanh --adam_beta 0.95   --poly_num_sample 1 --poly_order 1 --poly_coef 0.1

python train_cnf.py --data cifar10 --dims 64,64,64 --strides 1,1,1,1 --num_blocks 2 --layer_type concat --multiscale True --rademacher True --save experiments/cifar10 --adjoint --nonlinearity tanh --adam_beta 0.95 --poly_num_sample 1 --poly_order 1 --poly_coef 0.1
