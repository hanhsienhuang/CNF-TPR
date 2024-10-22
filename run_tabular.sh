# FFJORD
python train_tabular.py --data power     --adjoint --nonlinearity tanh --nhidden 3 --hdim_factor 10 --num_blocks 5  --batch_size 10000 --test_batch_size 10000 --save experiments/tabular-power-ffjord/     --layer_type concat 

python train_tabular.py --data gas       --adjoint --nonlinearity tanh --nhidden 3 --hdim_factor 20 --num_blocks 5  --batch_size 1000  --test_batch_size 10000 --save experiments/tabular-gas-ffjord/       --layer_type concat

python train_tabular.py --data miniboone --adjoint --nonlinearity softplus --nhidden 2 --hdim_factor 20 --num_blocks 1  --batch_size 1000  --test_batch_size 10000 --save experiments/tabular-miniboone-ffjord/ --layer_type concat

python train_tabular.py --data hepmass   --adjoint --nonlinearity softplus --nhidden 2 --hdim_factor 10 --num_blocks 10 --batch_size 10000 --test_batch_size 10000 --save experiments/tabular-hepmass-ffjord/   --layer_type concat

python train_tabular.py --data bsds300   --adjoint --nonlinearity tanh --nhidden 3 --hdim_factor 20 --num_blocks 2  --batch_size 10000 --test_batch_size 10000 --save experiments/tabular-bsds300-ffjord/   --layer_type concatsquash --optim adamax


# Ours
python train_tabular.py --data power     --adjoint --nonlinearity tanh --nhidden 3 --hdim_factor 10 --num_blocks 5  --batch_size 10000 --test_batch_size 10000 --save experiments/tabular-power-ours/     --layer_type concat --poly_num_sample 2 --poly_order 1 --poly_coef 5 --atol 1e-4 --rtol 1e-4 --test_atol 1e-5 --test_rtol 1e-5

python train_tabular.py --data gas       --adjoint --nonlinearity tanh --nhidden 3 --hdim_factor 20 --num_blocks 5  --batch_size 1000  --test_batch_size 10000 --save experiments/tabular-gas-ours/       --layer_type concat --poly_num_sample 2 --poly_order 1 --poly_coef 5 --atol 1e-4 --rtol 1e-4 --test_atol 1e-5 --test_rtol 1e-5

python train_tabular.py --data hepmass   --adjoint --nonlinearity softplus --nhidden 2 --hdim_factor 10 --num_blocks 10 --batch_size 10000 --test_batch_size 10000 --save experiments/tabular-hepmass-ours/   --layer_type concat --poly_num_sample 2 --poly_order 1 --poly_coef 5 --atol 1e-4 --rtol 1e-4 --test_atol 1e-5 --test_rtol 1e-5

python train_tabular.py --data miniboone --adjoint --nonlinearity softplus --nhidden 2 --hdim_factor 20 --num_blocks 1  --batch_size 1000  --test_batch_size 10000 --save experiments/tabular-miniboone-ours/ --layer_type concat --poly_num_sample 2 --poly_order 1 --poly_coef 5 --atol 1e-4 --rtol 1e-4 --test_atol 1e-5 --test_rtol 1e-5

python train_tabular.py --data bsds300   --adjoint --nonlinearity tanh --nhidden 3 --hdim_factor 20 --num_blocks 2  --batch_size 10000 --test_batch_size 10000 --save experiments/tabular-bsds300-ours/   --layer_type concatsquash --poly_num_sample 2 --poly_order 1 --poly_coef 5 --atol 1e-4 --rtol 1e-4 --test_atol 1e-5 --test_rtol 1e-5
