import io
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from rns.constant import W, H

# Plotter functions
PLOT_FUNCS = {}
def register_plotter(func):
    PLOT_FUNCS[func.__name__] = func
    def func_wrapper(images, **conv_kwargs):
        return func(images, **conv_kwargs)
    return func_wrapper

def plot(mode, vals, FLAGS, itr=0, save=True, return_buf=False, show=False):
    func = PLOT_FUNCS[mode]
    path = func(vals, FLAGS, itr=itr)

    buf = None
    if save:
        plt.savefig(path)
    if return_buf:
        buf = io.BytesIO()
        plt.savefig(buf)
        buf.seek(0)
    if show:
        plt.show()

    plt.close()
    return buf

@register_plotter
def arr(arr, FLAGS, itr=None):
    plt.imshow(arr, cmap='binary')

@register_plotter
def in_out_vae(vals, FLAGS, itr=0):
    vae_title = '{}-vae.png'.format(itr)
    os.makedirs(FLAGS['plot_path'], exist_ok=True)
    vae_path = os.path.join(FLAGS['plot_path'], vae_title) 
    fig, (ax1, ax2) = plt.subplots(1,2)
    ax1.imshow(vals['img1'])#, cmap='binary')
    ax2.imshow(vals['img2'])#, cmap='binary')
    return vae_path

@register_plotter
def contour(vals, FLAGS, itr=0):
    X, Y, Z, state = vals['X'], vals['Y'], vals['Z'], vals['state']
    prob_title = '{}-prob.png'.format(itr)
    os.makedirs(FLAGS['plot_path'], exist_ok=True)
    prob_path = os.path.join(FLAGS['plot_path'], prob_title) 
    plt.contour(X,Y,Z[:,:,0])
    plt.scatter(state[0,:,0], state[0,:,1])
    plt.title(prob_title)
    return prob_path

@register_plotter
def samples(vals, FLAGS, itr=0):
    samples = vals['samples']
    sample_title = '{}-sample.png'.format(itr)
    sample_path = os.path.join(FLAGS['plot_path'], sample_title) 

    sns.jointplot(samples[:,0,0], samples[:,0,1], kind='hex', color='#4cb391', xlim=(-1.0,1.0), ylim=(-1.0,1.0))
    return sample_path

@register_plotter
def shapes(vals, FLAGS, itr=None):
    dg = vals['dg']
    ax = plt.gca(aspect='equal', xlim=W, ylim=H)
    rect = mpatches.Rectangle((0,0), W, H, color='C0')
    ax.add_patch(rect)

    objs = dg.__next__()
    
    for o in objs['shapes']:
        o.plot(ax)

