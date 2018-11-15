import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from rns.constant import W, H


def plot_arr(arr):
    plt.imshow(arr, cmap='binary')
    plt.show()

def plot_in_out_vae(img1, img2):
    fig, (ax1, ax2) = plt.subplots(2,1)
    ax1.imshow(img1)
    ax2.imshow(img2)
    plt.show()

def plot_contour(curr_state, X, Y, Z, FLAGS, i=None):
    prob_title = '{}-prob.png'.format(i)
    os.makedirs(FLAGS['plot_path'], exist_ok=True)
    prob_path = os.path.join(FLAGS['plot_path'], prob_title) 

    plt.contour(X,Y,Z[:,:,0])
    plt.scatter(curr_state[0,:,0], curr_state[0,:,1])
    plt.title(prob_title)
    plt.savefig(prob_path)
    plt.clf()

def plot_samples(samples, FLAGS, i=None):
    sample_title = '{}-sample.png'.format(i)
    sample_path = os.path.join(FLAGS['plot_path'], sample_title) 

    sns.jointplot(samples[:,0,0], samples[:,0,1], kind='hex', color='#4cb391', xlim=(-1.0,1.0), ylim=(-1.0,1.0))
    plt.savefig(sample_path)
    plt.clf()


def plot_shapes(data_generator, FLAGS):
    ax = plt.gca(aspect='equal', xlim=W, ylim=H)
    rect = mpatches.Rectangle((0,0), W, H, color='C0')
    ax.add_patch(rect)

    objs = data_generator.__next__()
    
    for o in objs['shapes']:
        o.plot(ax)
    
    plt.show()


