for data in 'rings' '8gaussians' 'checkerboard' 'circles' 'moons' 'pinwheel' 'swissroll' '2spirals' ; do
    python train_toy.py --data $data --save experiments/toy-${data}-concat-sample2-acc2-0.1 --num_sample 2 --coef_acc 0.1 --solver dopri5 --adjoint --niters 10000 --viz_freq 10000 --layer_type concat
    python train_toy.py --data $data --save experiments/toy-${data}-concat-sample2          --num_sample 2                --solver dopri5 --adjoint --niters 10000 --viz_freq 10000 --layer_type concat
    python train_toy.py --data $data --save experiments/toy-${data}-concat-acc2-0.1                        --coef_acc 0.1 --solver dopri5 --adjoint --niters 10000 --viz_freq 10000 --layer_type concat
    python train_toy.py --data $data --save experiments/toy-${data}-concat                                                --solver dopri5 --adjoint --niters 10000 --viz_freq 10000 --layer_type concat
done
