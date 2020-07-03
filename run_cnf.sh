python train_cnf.py --data mnist   --dims 64,64,64 --strides 1,1,1,1 --num_blocks 2 --layer_type concat --multiscale True --rademacher True --save experiments/mnist-ffjord --adjoint --nonlinearity softplus

python train_cnf.py --data cifar10 --dims 64,64,64 --strides 1,1,1,1 --num_blocks 2 --layer_type concat --multiscale True --rademacher True --save experiments/cifar10-ffjord --adjoint --nonlinearity softplus

# TOL 4
python train_cnf.py --data mnist   --dims 64,64,64 --strides 1,1,1,1 --num_blocks 2 --layer_type concat --multiscale True --rademacher True --save experiments/mnist-spl2-ord1-coef5-softplus-tol4 --adjoint --nonlinearity softplus --poly_num_sample 2 --poly_order 1 --poly_coef 5 --atol 1e-4 --rtol 1e-4 --test_atol 1e-5 --test_rtol 1e-5
python train_cnf.py --data cifar10 --dims 64,64,64 --strides 1,1,1,1 --num_blocks 2 --layer_type concat --multiscale True --rademacher True --save experiments/cifar10-spl2-ord1-coef5-softplus-tol4 --adjoint --nonlinearity softplus --poly_num_sample 2 --poly_order 1 --poly_coef 5 --atol 1e-4 --rtol 1e-4 --test_atol 1e-5 --test_rtol 1e-5
