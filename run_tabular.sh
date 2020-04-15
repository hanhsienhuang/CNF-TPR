python train_tabular.py --data power     --adjoint --nonlinearity tanh --nhidden 3 --hdim_factor 10 --num_blocks 5  --batch_size 10000 --test_batch_size 10000 --save experiments/tabular-power/     --layer_type concat --adam_beta 0.95 --poly_num_sample 1 --poly_order 1 --poly_coef 0.1

python train_tabular.py --data gas       --adjoint --nonlinearity tanh --nhidden 3 --hdim_factor 20 --num_blocks 5  --batch_size 1000  --test_batch_size 10000 --save experiments/tabular-gas/       --layer_type concat --adam_beta 0.95 --poly_num_sample 1 --poly_order 1 --poly_coef 0.1

python train_tabular.py --data hepmass   --adjoint --nonlinearity tanh --nhidden 2 --hdim_factor 10 --num_blocks 10 --batch_size 10000 --test_batch_size 10000 --save experiments/tabular-hepmass/   --layer_type concat --adam_beta 0.95 --poly_num_sample 1 --poly_order 1 --poly_coef 0.1

python train_tabular.py --data miniboone --adjoint --nonlinearity tanh --nhidden 2 --hdim_factor 20 --num_blocks 1  --batch_size 1000  --test_batch_size 10000 --save experiments/tabular-miniboone/ --layer_type concat --adam_beta 0.95 --poly_num_sample 1 --poly_order 1 --poly_coef 0.1

python train_tabular.py --data bsds300   --adjoint --nonlinearity tanh --nhidden 3 --hdim_factor 20 --num_blocks 2  --batch_size 10000 --test_batch_size 10000 --save experiments/tabular-bsds300/   --layer_type concat --adam_beta 0.95 --poly_num_sample 1 --poly_order 1 --poly_coef 0.1
