python train_vae_flow.py --dataset freyfaces --flow cnf_rank --rank 20 --dims 512-512   --num_blocks 2 --nonlinearity tanh --out_dir experiments/vae-freyfaces --poly_num_sample 1 --poly_coef 10 --poly_order 1

python train_vae_flow.py --dataset caltech   --flow cnf_rank --rank 20 --dims 2048      --num_blocks 1 --nonlinearity tanh --out_dir experiments/vae-caltech   --poly_num_sample 1 --poly_coef 10 --poly_order 1

python train_vae_flow.py --dataset mnist     --flow cnf_rank --rank 64 --dims 1024-1024 --num_blocks 2 --nonlinearity tanh --out_dir experiments/vae-mnist     --poly_num_sample 1 --poly_coef 10 --poly_order 1

python train_vae_flow.py --dataset omniglot  --flow cnf_rank --rank 20 --dims 512-512   --num_blocks 5 --nonlinearity tanh --out_dir experiments/vae-omniglot  --poly_num_sample 1 --poly_coef 10 --poly_order 1


