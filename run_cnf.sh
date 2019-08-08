python train_cnf.py --data mnist   --dims 64,64,64 --strides 1,1,1,1 --num_blocks 2 --layer_type concat --multiscale True --rademacher True --save experiments/mnist-sample2-acc2-0.1-adambeta0.95  --solver dopri5 --adjoint --nonlinearity tanh --num_sample 2 --coef_acc 0.1 

python train_cnf.py --data cifar10 --dims 64,64,64 --strides 1,1,1,1 --num_blocks 2 --layer_type concat --multiscale True --rademacher True --save experiments/cifar10-sample2-acc2-0.1-adambeta0.95  --solver dopri5 --adjoint --nonlinearity tanh --num_sample 2 --coef_acc 0.1
