import torch
from train_misc import build_model_tabular
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import math
from matplotlib.collections import LineCollection

name_format = "experiments/toy-{}-concat-sampacc-samp1-secondsmooth-0.01/checkpt.pth"

def standard_normal_logprob(z):
    logZ = -0.5 * math.log(2 * math.pi)
    return logZ - z.pow(2) / 2

def plot(ax, filename, title):
    checkpt = torch.load(filename, map_location=lambda storage, loc: storage)
    model = build_model_tabular(checkpt['args'], 2, [])
    model.load_state_dict(checkpt["state_dict"])
    model.eval()

    ax.set_aspect(aspect="equal")
    side = np.linspace(-3, 3, 10)
    xx, yy = np.meshgrid(side, side)
    z = np.stack([xx.reshape(-1), yy.reshape(-1)], -1)
    z = torch.Tensor(z)

    num_steps = 20
    t = torch.linspace(0, 0.5, steps = num_steps)
    #t = torch.linspace(0, 1, steps = num_steps)
    ys = model.chain[0](z, reverse = True, integration_times = t)[0]
    ys = ys.detach().numpy()

    for i in range(z.shape[0]):
        segments = np.stack((ys[:-1,i], ys[1:,i]), 1)
        lc = LineCollection(segments, cmap="magma")
        lc.set_array(t.detach().numpy())
        line = ax.add_collection(lc)

    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_title(title)
    ax.set_xticks([],[])
    ax.set_yticks([],[])
    ax.invert_yaxis()
    return line

if __name__ == "__main__":
    data = [
    'swissroll',
    '8gaussians',
    'pinwheel',
    'circles',
    'moons',
    '2spirals',
    'checkerboard',
    'rings',
    ]
    fig, axes = plt.subplots(3, 3, figsize = (3.3, 4), sharex = True, sharey = True)
    fig.subplots_adjust(0,-0.03,1,0.97, wspace=0.05, hspace=0.)
    axes = axes.reshape(-1)
    for i, (d, ax) in enumerate(zip(data, axes)):
        filename = name_format.format(d)
        line = plot(ax, filename, str(i+1))

    axes[-1].axis('off')
    #cbar = fig.colorbar(line, ticks = [0,0.5])
    #cbar.ax.set_yticklabels(["Start", "End"])
    #cbar.ax.set_ylabel("t")
    plt.savefig("../plot/2d-traj-ours.pdf")
    plt.show()
