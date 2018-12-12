#!/usr/bin/env python3
import os
import itertools
import numpy as np
import tensorflow as tf
import tqdm
from define_flags import FLAGS
from rns.data import normalize, subsample_postbatch, to_float, data_generator, Dataset
from rns.viz import plot
from rns.models import RNModel, ConvAE, ConvVAE, ConvRN_VAE, State2ImageVAE, ConvMDN, SingletonModel

# TODO: need to add mechanism for masking out extra objects so that every batch can have a fixed number of inputs.
# at the moment, there can be a difference between number of objects in batch and that shown in the image
# TODO: add summaries visualizing the distributions
# TODO: annealing learning rate
# TODO: try RN network for images
# TODO: optimize RN model
# TODO: add a global step for adam optimizer
# TODO: try leaky relu
# TODO: add timing

# TODO: combine these into nicer single function, with standard model interface that can get run
numvars = lambda : print("\nNUM VARIABLES", np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]), "\n")

def main():
    train_ds = Dataset(FLAGS)
    Model = {'rn': RNModel, 'conv_mdn': ConvMDN, 'single': SingletonModel}[FLAGS['mode']]
    model = Model(train_ds.state, FLAGS)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter(FLAGS['log_path'] + '/train', sess.graph)
    numvars()

    try:
        pbar = tqdm.tqdm(total=FLAGS['tqdm_n'])
        for i in itertools.count(start=1):
            train_vals = sess.run(model.train_vals)
            pbar.update()
            if i % 100 == 0:
                eval_vals = sess.run(model.eval_vals)
                #plot_arr(curr_state['image'][0][...,0])
                train_writer.add_summary(eval_vals['summary'], global_step=i)
                subD = {key: eval_vals[key] for key in ['state', 'X', 'Y', 'Z']}
                plots = plot('contour', subD, FLAGS, itr=i, return_buf=True)
                plot_summary = sess.run(model.plot_summaries, {model.pred_plot_ph: plots.getvalue()})
                train_writer.add_summary(plot_summary, global_step=i)

                #print('scales: min: {} max: {} mean: {} median: {}'.format(np.min(scales[0]), np.max(scales[0]), np.mean(scales[0]), np.median(scales[0])))
                logits = eval_vals['logits']
                #print('logits: min: {} max: {} median: {}'.format(np.min(logits[0]), np.max(logits[0]), np.median(logits[0])))
                #print('i = {}, loss = {}'.format(i, eval_vals['loss']))
            if i % FLAGS['tqdm_n'] == 0:
                pbar.close()
                pbar = tqdm.tqdm(total=FLAGS['tqdm_n'])

            if i >= 5e4:
                exit()

    except KeyboardInterrupt:
        import ipdb; ipdb.set_trace()


def vae_main():
    train_ds = Dataset(FLAGS)

    Model = {'vae': ConvVAE, 'vae_rn': ConvRN_VAE, 'rn2img': State2ImageVAE}[FLAGS['mode']]
    model = Model(train_ds.state, FLAGS)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    numvars()

    train_writer = tf.summary.FileWriter(FLAGS['log_path'] + '/train', sess.graph)

    pbar = tqdm.tqdm(total=FLAGS['tqdm_n'])
    for i in itertools.count(start=1):
        train_vals = sess.run(model.train_vals)
        pbar.update()

        if i % 100 == 0:
            eval_vals = sess.run(model.eval_vals)
            train_writer.add_summary(eval_vals['summary'], global_step=i)
            #plot_arr(curr_state['image'][0][...,0])
            vals = {'img1': eval_vals['state']['image'][0][...,0], 'img2': eval_vals['samples'][0][0][...,0]}
            plot('in_out_vae', vals, FLAGS, itr=i)
            #print('scales: min: {} max: {} mean: {} median: {}'.format(np.min(scales[0]), np.max(scales[0]), np.mean(scales[0]), np.median(scales[0])))
            print('i = {}, loss = {}'.format(i, eval_vals['loss']))
            train_writer.flush()

        if i % FLAGS['tqdm_n'] == 0:
            pbar.close()
            pbar = tqdm.tqdm(total=FLAGS['tqdm_n'])
        if i >= 5e4:
            exit()

def ae_main():
    train_ds = Dataset(FLAGS)
    model = ConvAE(train_ds.state, FLAGS)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    numvars()

    train_writer = tf.summary.FileWriter(FLAGS['log_path'] + '/train', sess.graph)

    pbar = tqdm.tqdm(total=FLAGS['tqdm_n'])
    for i in itertools.count(start=1):
        train_vals = sess.run(model.train_vals)
        pbar.update()
        if i % 100 == 0:
            eval_vals = sess.run(model.eval_vals)
            train_writer.add_summary(eval_vals['summary'], global_step=i)
            train_writer.flush()
        if i % FLAGS['tqdm_n'] == 0:
            pbar.close()
            pbar = tqdm.tqdm(total=FLAGS['tqdm_n'])
        if i >= 5e4:
            exit()

if __name__ == "__main__":
    if FLAGS['plot_shapes']:
        #while True:
        #    plot_shapes(data_generator(FLAGS['num_shapes']), FLAGS)
        while True:
            dg = data_generator(FLAGS['num_shapes'], FLAGS['samplers'])
            plot('arr', dg.__next__()['image'][...,0], FLAGS)

    {'vae': vae_main, 'ae': ae_main, 'rn': main, 'vae_rn': vae_main, 'rn2img': vae_main, 'conv_mdn': main, 'single': main}[FLAGS['mode']]()
