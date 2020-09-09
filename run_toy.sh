for data in 'rings' '8gaussians' 'checkerboard' 'circles' 'moons' 'pinwheel' 'swissroll' '2spirals' ; do
    python train_toy.py --data $data --save experiments/toy-${data} --adjoint --niters 10000 --viz_freq 10000 --layer_type concat --poly_num_sample 2 --poly_order 1 --poly_coef 5 --atol 1e-4 --rtol 1e-4 --test_atol 1e-5 --test_rtol 1e-5
done
