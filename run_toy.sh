for data in '2spirals' '8gaussians' 'checkerboard' 'circles' 'moons' 'pinwheel' 'rings' 'swissroll' ; do
    python train_toy.py --data $data --save /tmp/experiments/toy-${data}-samp1-acc2-0.1 --solver dopri5  --num_steps 1 --adjoint --acc2 0.1 --niters 10000 --viz_freq 5000
done
