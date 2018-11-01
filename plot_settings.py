import matplotlib.pyplot as plt

plt.rc('font', family='serif', serif='Computer Modern Roman')
plt.rc('text', usetex=True)
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
plt.rc('axes', labelsize=12)
plt.rc('legend', fontsize=12)

default_width = 4
default_height = default_width/1.618

def savefig(fig, loc, width=default_width, height=default_height):
    fig.set_size_inches(width, height)
    fig.savefig(loc)
