for data in 'rings' '8gaussians' 'checkerboard' 'circles' 'moons' 'pinwheel' 'swissroll' '2spirals' ; do
    python train_toy.py --data $data --save experiments/toy-${data} --num_sample 2 --coef_acc 0.1 --adjoint --niters 10000 --viz_freq 10000 --layer_type concat
done
