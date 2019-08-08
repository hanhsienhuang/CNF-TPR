python train_tabular.py --data power     --adjoint --nonlinearity tanh --nhidden 3 --hdim_factor 10 --num_blocks 5  --batch_size 10000 --test_batch_size 10000 --solver dopri5 --save experiments/tabular-power-concat-sample2-acc2-0.1-adambeta0.95/     --num_sample 2 --coef_acc 0.1 --layer_type concat

python train_tabular.py --data gas       --adjoint --nonlinearity tanh --nhidden 3 --hdim_factor 20 --num_blocks 5  --batch_size 1000  --test_batch_size 10000 --solver dopri5 --save experiments/tabular-gas-concat-sample2-acc2-0.1-adambeta0.95/       --num_sample 2 --coef_acc 0.1 --layer_type concat

python train_tabular.py --data hepmass   --adjoint --nonlinearity tanh --nhidden 2 --hdim_factor 10 --num_blocks 10 --batch_size 10000 --test_batch_size 10000 --solver dopri5 --save experiments/tabular-hepmass-concat-sample2-acc2-0.1-adambeta0.95/   --num_sample 2 --coef_acc 0.1 --layer_type concat

python train_tabular.py --data miniboone --adjoint --nonlinearity tanh --nhidden 2 --hdim_factor 20 --num_blocks 1  --batch_size 1000  --test_batch_size 10000 --solver dopri5 --save experiments/tabular-miniboone-concat-sample2-acc2-0.1-adambeta0.95/ --num_sample 2 --coef_acc 0.1 --layer_type concat 

python train_tabular.py --data bsds300   --adjoint --nonlinearity tanh --nhidden 3 --hdim_factor 20 --num_blocks 2  --batch_size 10000 --test_batch_size 10000 --solver dopri5 --save experiments/tabular-bsds300-concat-sample2-acc2-0.1-adambeta0.95/   --num_sample 2 --coef_acc 0.1 --layer_type concat 


