for data in 'rings' '8gaussians' 'checkerboard' 'circles' 'moons' 'pinwheel' 'swissroll' '2spirals' ; do
    python train_toy.py --data $data --save experiments/toy-${data} --adjoint --niters 10000 --viz_freq 10000 --layer_type concat --poly_coef 0.1 --poly_num_sample 1 --poly_order 1
done
