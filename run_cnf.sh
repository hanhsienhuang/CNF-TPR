#python train_cnf.py --data mnist   --dims 64,64,64 --strides 1,1,1,1 --num_blocks 2 --layer_type concat --multiscale True --rademacher True --save experiments/mnist-sample2-acc2-0.1-adambeta0.95  --solver dopri5 --adjoint --nonlinearity tanh --num_sample 2 --coef_acc 0.1 --time 464978.685

#python train_cnf.py --data cifar10 --dims 64,64,64 --strides 1,1,1,1 --num_blocks 2 --layer_type concat --multiscale True --rademacher True --save experiments/cifar10-sample2-acc2-0.1-adambeta0.95  --solver dopri5 --adjoint --nonlinearity tanh --num_sample 2  --coef_acc 0.1 --time 534248.804


## With sampling without error control
#python train_cnf.py --data mnist   --dims 64,64,64 --strides 1,1,1,1 --num_blocks 2 --layer_type concat --multiscale True --rademacher True --save experiments/mnist-sample2-adambeta0.95  --solver dopri5 --adjoint --nonlinearity tanh --num_sample 2 --time 464978.685
#
#python train_cnf.py --data cifar10 --dims 64,64,64 --strides 1,1,1,1 --num_blocks 2 --layer_type concat --multiscale True --rademacher True --save experiments/cifar10-sample2-adambeta0.95  --solver dopri5 --adjoint --nonlinearity tanh --num_sample 2 --time 534248.804


#python train_cnf.py --data mnist   --dims 64,64,64 --strides 1,1,1,1 --num_blocks 2 --layer_type concat --multiscale True --rademacher True --save experiments/mnist-acc2-0.1-adambeta0.95  --solver dopri5 --adjoint --nonlinearity tanh --coef_acc 0.1 --time 464978.685
#
#python train_cnf.py --data cifar10 --dims 64,64,64 --strides 1,1,1,1 --num_blocks 2 --layer_type concat --multiscale True --rademacher True --save experiments/cifar10-acc2-0.1-adambeta0.95  --solver dopri5 --adjoint --nonlinearity tanh  --coef_acc 0.1 --time 534248.804 --test_batch_size 200 --batch_size


#python train_cnf.py --data mnist   --dims 64,64,64 --strides 1,1,1,1 --num_blocks 2 --layer_type concat --multiscale True --rademacher True --save experiments/mnist  --solver dopri5 --adjoint --nonlinearity tanh --time 464978.685

python train_cnf.py --data cifar10 --dims 64,64,64 --strides 1,1,1,1 --num_blocks 2 --layer_type concat --multiscale True --rademacher True --save experiments/cifar10  --solver dopri5 --adjoint --nonlinearity tanh  --time 534248.804 
