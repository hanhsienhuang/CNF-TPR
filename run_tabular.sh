python train_tabular.py --data power     --adjoint --nonlinearity tanh --nhidden 3 --hdim_factor 10 --num_blocks 5  --batch_size 10000 --test_batch_size 10000 --solver rk4 --num_steps 4 --test_solver dopri5 --acc_smooth 1 --save experiments/tabular-power/
python train_tabular.py --data gas       --adjoint --nonlinearity tanh --nhidden 3 --hdim_factor 20 --num_blocks 5  --batch_size 1000  --test_batch_size 10000 --solver rk4 --num_steps 4 --test_solver dopri5 --acc_smooth 1 --save experiments/tabular-gas/
python train_tabular.py --data hepmass   --adjoint --nonlinearity tanh --nhidden 2 --hdim_factor 10 --num_blocks 10 --batch_size 10000 --test_batch_size 10000 --solver rk4 --num_steps 4 --test_solver dopri5 --acc_smooth 1 --save experiments/tabular-hepmass/
python train_tabular.py --data miniboone --adjoint --nonlinearity tanh --nhidden 2 --hdim_factor 20 --num_blocks 1  --batch_size 1000  --test_batch_size 10000 --solver rk4 --num_steps 4 --test_solver dopri5 --acc_smooth 1 --save experiments/tabular-miniboone/
python train_tabular.py --data bsds300   --adjoint --nonlinearity tanh --nhidden 3 --hdim_factor 20 --num_blocks 2  --batch_size 10000 --test_batch_size 10000 --solver rk4 --num_steps 8 --test_solver dopri5 --acc_smooth 1 --save experiments/tabular-bsds300/


