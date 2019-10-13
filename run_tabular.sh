#python train_tabular.py --data power     --adjoint --nonlinearity tanh --nhidden 3 --hdim_factor 10 --num_blocks 5  --batch_size 10000 --test_batch_size 10000 --solver dopri5 --save experiments/tabular-power-concat-sample2-acc2-0.1-adambeta0.95/     --num_sample 2 --coef_acc 0.1 --layer_type concat --time 122947.895
#
#python train_tabular.py --data gas       --adjoint --nonlinearity tanh --nhidden 3 --hdim_factor 20 --num_blocks 5  --batch_size 1000  --test_batch_size 10000 --solver dopri5 --save experiments/tabular-gas-concat-sample2-acc2-0.1-adambeta0.95/       --num_sample 2 --coef_acc 0.1 --layer_type concat --time 128089.204
#
#python train_tabular.py --data hepmass   --adjoint --nonlinearity tanh --nhidden 2 --hdim_factor 10 --num_blocks 10 --batch_size 10000 --test_batch_size 10000 --solver dopri5 --save experiments/tabular-hepmass-concat-sample2-acc2-0.1-adambeta0.95/   --num_sample 2 --coef_acc 0.1 --layer_type concat --time 158430.819
#
#python train_tabular.py --data miniboone --adjoint --nonlinearity tanh --nhidden 2 --hdim_factor 20 --num_blocks 1  --batch_size 1000  --test_batch_size 10000 --solver dopri5 --save experiments/tabular-miniboone-concat-sample2-acc2-0.1-adambeta0.95/ --num_sample 2 --coef_acc 0.1 --layer_type concat --time 11618.082
#
#python train_tabular.py --data bsds300   --adjoint --nonlinearity tanh --nhidden 3 --hdim_factor 20 --num_blocks 2  --batch_size 10000 --test_batch_size 10000 --solver dopri5 --save experiments/tabular-bsds300-concat-sample2-acc2-0.1-adambeta0.95/   --num_sample 2 --coef_acc 0.1 --layer_type concat --time 212350.971


#### With sampling with error control
#python train_tabular.py --data power     --adjoint --nonlinearity tanh --nhidden 3 --hdim_factor 10 --num_blocks 5  --batch_size 10000 --test_batch_size 10000 --solver dopri5 --save experiments/tabular-power-concat-sample2-adambeta0.95/     --num_sample 2 --layer_type concat --time 122947.895
#
#python train_tabular.py --data gas       --adjoint --nonlinearity tanh --nhidden 3 --hdim_factor 20 --num_blocks 5  --batch_size 1000  --test_batch_size 10000 --solver dopri5 --save experiments/tabular-gas-concat-sample2-adambeta0.95/       --num_sample 2 --layer_type concat --time 128089.204
#
#python train_tabular.py --data hepmass   --adjoint --nonlinearity tanh --nhidden 2 --hdim_factor 10 --num_blocks 10 --batch_size 10000 --test_batch_size 10000 --solver dopri5 --save experiments/tabular-hepmass-concat-sample2-adambeta0.95/   --num_sample 2 --layer_type concat --time 158430.819
#
#python train_tabular.py --data miniboone --adjoint --nonlinearity tanh --nhidden 2 --hdim_factor 20 --num_blocks 1  --batch_size 1000  --test_batch_size 10000 --solver dopri5 --save experiments/tabular-miniboone-concat-sample2-adambeta0.95/ --num_sample 2 --layer_type concat --time 11618.082
#
#python train_tabular.py --data bsds300   --adjoint --nonlinearity tanh --nhidden 3 --hdim_factor 20 --num_blocks 2  --batch_size 10000 --test_batch_size 10000 --solver dopri5 --save experiments/tabular-bsds300-concat-sample2-adambeta0.95/   --num_sample 2 --layer_type concat --time 212350.971


#python train_tabular.py --data power     --adjoint --nonlinearity tanh --nhidden 3 --hdim_factor 10 --num_blocks 5  --batch_size 10000 --test_batch_size 10000 --solver dopri5 --save experiments/tabular-power-concat-acc2-0.1-adambeta0.95/     --coef_acc 0.1 --layer_type concat --time 122947.895
#
#python train_tabular.py --data gas       --adjoint --nonlinearity tanh --nhidden 3 --hdim_factor 20 --num_blocks 5  --batch_size 1000  --test_batch_size 10000 --solver dopri5 --save experiments/tabular-gas-concat-acc2-0.1-adambeta0.95/       --coef_acc 0.1 --layer_type concat --time 128089.204
#
#python train_tabular.py --data hepmass   --adjoint --nonlinearity tanh --nhidden 2 --hdim_factor 10 --num_blocks 10 --batch_size 10000 --test_batch_size 10000 --solver dopri5 --save experiments/tabular-hepmass-concat-acc2-0.1-adambeta0.95/   --coef_acc 0.1 --layer_type concat --time 158430.819
#
#python train_tabular.py --data miniboone --adjoint --nonlinearity tanh --nhidden 2 --hdim_factor 20 --num_blocks 1  --batch_size 1000  --test_batch_size 10000 --solver dopri5 --save experiments/tabular-miniboone-concat-acc2-0.1-adambeta0.95/ --coef_acc 0.1 --layer_type concat --time 11618.082
#
#python train_tabular.py --data bsds300   --adjoint --nonlinearity tanh --nhidden 3 --hdim_factor 20 --num_blocks 2  --batch_size 10000 --test_batch_size 10000 --solver dopri5 --save experiments/tabular-bsds300-concat-acc2-0.1-adambeta0.95/   --coef_acc 0.1 --layer_type concat --time 212350.971
#python train_tabular.py --data bsds300   --adjoint --nonlinearity tanh --nhidden 3 --hdim_factor 20 --num_blocks 2  --batch_size 10000 --test_batch_size 10000 --solver dopri5 --save experiments/tabular-bsds300-concat-acc2-0.1-adambeta0.95/   --coef_acc 0.1 --layer_type concat --time 212350.971



#python train_tabular.py --data power     --adjoint --nonlinearity tanh --nhidden 3 --hdim_factor 10 --num_blocks 5  --batch_size 10000 --test_batch_size 10000 --solver dopri5 --save experiments/tabular-power-concat/     --layer_type concat --time 122947.895
#
#python train_tabular.py --data gas       --adjoint --nonlinearity tanh --nhidden 3 --hdim_factor 20 --num_blocks 5  --batch_size 1000  --test_batch_size 10000 --solver dopri5 --save experiments/tabular-gas-concat/       --layer_type concat --time 128089.204

python train_tabular.py --data hepmass   --adjoint --nonlinearity tanh --nhidden 2 --hdim_factor 10 --num_blocks 10 --batch_size 10000 --test_batch_size 10000 --solver dopri5 --save experiments/tabular-hepmass-concat/   --layer_type concat --time 158430.819

python train_tabular.py --data miniboone --adjoint --nonlinearity tanh --nhidden 2 --hdim_factor 20 --num_blocks 1  --batch_size 1000  --test_batch_size 10000 --solver dopri5 --save experiments/tabular-miniboone-concat/ --layer_type concat --time 11618.082

python train_tabular.py --data bsds300   --adjoint --nonlinearity tanh --nhidden 3 --hdim_factor 20 --num_blocks 2  --batch_size 10000 --test_batch_size 10000 --solver dopri5 --save experiments/tabular-bsds300-concat/   --layer_type concat --time 212350.971
