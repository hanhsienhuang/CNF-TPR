#python train_tabular.py --data power     --adjoint --nonlinearity tanh --nhidden 3 --hdim_factor 10 --num_blocks 5  --batch_size 10000 --test_batch_size 10000 --solver dopri5 --acc2 0.1 --save experiments/tabular-power-concat-samp1-acc2-0.1-early30/ --num_steps 1 --layer_type concat
### python train_tabular.py --data power     --adjoint --nonlinearity tanh --nhidden 3 --hdim_factor 10 --num_blocks 5  --batch_size 10000 --test_batch_size 10000 --solver dopri5 --acc2 0.1 --save /tmp/test --num_steps 1 --layer_type concat --evaluate --resume experiments/tabular-power-concat-samp1-acc2-0.1-early30/checkpt.pth

#python train_tabular.py --data gas       --adjoint --nonlinearity tanh --nhidden 3 --hdim_factor 20 --num_blocks 5  --batch_size 1000  --test_batch_size 10000 --solver dopri5 --acc2 0.1 --save experiments/tabular-gas-blend-samp1-acc2-0.1-early30/ --num_steps 1 --layer_type blend
###python train_tabular.py --data gas       --adjoint --nonlinearity tanh --nhidden 3 --hdim_factor 20 --num_blocks 5  --batch_size 1000  --test_batch_size 10000 --solver dopri5 --acc2 0.1 --save /tmp/experiments/testgas/ --num_steps 1 --layer_type concat --resume experiments/tabular-gas-concat-samp1-acc2-0.1-early30/checkpt.pth --evaluate

#python train_tabular.py --data hepmass   --adjoint --nonlinearity tanh --nhidden 2 --hdim_factor 10 --num_blocks 10 --batch_size 10000 --test_batch_size 10000 --solver dopri5 --acc2 0.1 --save experiments/tabular-hepmass/ --num_steps 1

#python train_tabular.py --data miniboone --adjoint --nonlinearity tanh --nhidden 2 --hdim_factor 20 --num_blocks 1  --batch_size 1000  --test_batch_size 10000 --solver dopri5 --acc2 0.1 --save /tmp/experiments/tabular-miniboone-blend-samp1-acc2-0.1-early30/ --num_steps 1 --layer_type blend

python train_tabular.py --data bsds300   --adjoint --nonlinearity tanh --nhidden 3 --hdim_factor 20 --num_blocks 2  --batch_size 10000 --test_batch_size 10000 --solver dopri5 --acc2 0.1 --save experiments/tabular-bsds300-step4-acc2-0.1-adambeta0.95/ --num_steps 4


