python train_vae_flow.py --dataset mnist     --flow cnf_rank --rank 64 --dims 1024-1024 --num_blocks 2 --nonlinearity softplus --out_dir experiments/vae-mnist-ffjord  --optim adamax

python train_vae_flow.py --dataset omniglot  --flow cnf_rank --rank 20 --dims 512-512   --num_blocks 5 --nonlinearity softplus --out_dir experiments/vae-omniglot-ffjord  --optim adamax

python train_vae_flow.py --dataset freyfaces --flow cnf_rank --rank 20 --dims 512-512   --num_blocks 2 --nonlinearity softplus --out_dir experiments/vae-freyfaces-ffjord --optim adamax

python train_vae_flow.py --dataset caltech   --flow cnf_rank --rank 20 --dims 2048      --num_blocks 1 --nonlinearity tanh --out_dir experiments/vae-caltech-ffjord   --optim adamax

# TOL 4
python train_vae_flow.py --dataset mnist     --flow cnf_rank --rank 64 --dims 1024-1024 --num_blocks 2 --nonlinearity softplus --out_dir experiments/vae-mnist-spl2-ord1-coef5-act-adamax-tol4     --poly_num_sample 2 --poly_order 1 --poly_coef 5 --optim adamax --atol 1e-4 --rtol 1e-4 --test_atol 1e-5 --test_rtol 1e-5

python train_vae_flow.py --dataset omniglot  --flow cnf_rank --rank 20 --dims 512-512   --num_blocks 5 --nonlinearity softplus --out_dir experiments/vae-omniglot-spl2-ord1-coef5-act-adamax-tol4  --poly_num_sample 2 --poly_order 1 --poly_coef 5 --optim adamax --atol 1e-4 --rtol 1e-4 --test_atol 1e-5 --test_rtol 1e-5

python train_vae_flow.py --dataset freyfaces --flow cnf_rank --rank 20 --dims 512-512   --num_blocks 2 --nonlinearity softplus --out_dir experiments/vae-freyfaces-spl2-ord1-coef5-act-adamax-tol4 --poly_num_sample 2 --poly_order 1 --poly_coef 5 --optim adamax --atol 1e-4 --rtol 1e-4 --test_atol 1e-5 --test_rtol 1e-5 

python train_vae_flow.py --dataset caltech   --flow cnf_rank --rank 20 --dims 2048      --num_blocks 1 --nonlinearity tanh --out_dir experiments/vae-caltech-spl2-ord1-coef5-act-adamax-tol4   --poly_num_sample 2 --poly_order 1 --poly_coef 5 --optim adamax  --atol 1e-4 --rtol 1e-4 --test_atol 1e-5 --test_rtol 1e-5

