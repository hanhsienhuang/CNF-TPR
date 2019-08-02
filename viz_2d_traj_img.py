import viz_2d_traj
import argparse 
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('file', type=str)
args = parser.parse_args()


plt.figure(figsize=[4,4])
ax = plt.gca()
viz_2d_traj.plot(ax, args.file, "")
plt.tight_layout()
plt.show()
