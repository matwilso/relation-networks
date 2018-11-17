#!/usr/bin/env python3
import os
import itertools
import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd
import tqdm
from define_flags import FLAGS
from rns.data import normalize, subsample_postbatch, to_float, data_generator, Dataset
from rns.viz import plot
from rns.constant import W, H, R, IMAGE_SHAPE

# TODO: need to add mechanism for masking out extra objects so that every batch can have a fixed number of inputs.
# at the moment, there can be a difference between number of objects in batch and that shown in the image
# TODO: add summaries visualizing the distributions
# TODO: annealing learning rate
# TODO: try RN network for images
# TODO: optimize RN model
# TODO: add a global step for adam optimizer
# TODO: try leaky relu
# TODO: add timing


def pack_images(images, rows, cols):
    """Helper utility to make a field of images."""
    shape = tf.shape(images)
    width = shape[-3]
    height = shape[-2]
    depth = shape[-1]
    images = tf.reshape(images, (-1, width, height, depth))
    batch = tf.shape(images)[0]
    rows = tf.minimum(rows, batch)
    cols = tf.minimum(batch // rows, cols)
    images = images[:rows * cols]
    images = tf.reshape(images, (rows, cols, width, height, depth))
    images = tf.transpose(images, [0, 2, 1, 3, 4])
    images = tf.reshape(images, [1, rows * width, cols * height, depth])
    return images

def image_tile_summary(name, tensor, rows=8, cols=8):
    tf.summary.image(name, pack_images(tensor, rows, cols), max_outputs=3)

def cartesian_product(a,b):
    a, b = a[None, :, None], b[:, None, None]
    prod = tf.concat([b + tf.zeros_like(a), tf.zeros_like(b) + a], axis = 2)
    #new_shape = tf.stack([-1, tf.shape(cartesian_product)[-1]])
    #cartesian_product = tf.reshape(cartesian_product, new_shape)
    prod = tf.reshape(prod, [-1])
    return prod

class RNModel(object):
    def relation(self, oij, scope='g'):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE) as scope:
            #out = tf.concat([objects[ij[0]],objects[ij[0]]])
            out = oij
            out = tf.layers.dense(out, 128, activation=tf.nn.relu)
            out = tf.layers.dense(out, 128, activation=tf.nn.relu)
            out = tf.layers.dense(out, 128, activation=None)
        return out

    def f(self, g_sum):
        with tf.variable_scope('f') as scope:
            out = g_sum
            out = tf.layers.dense(out, 128, activation=tf.nn.relu)
            out = tf.layers.dense(out, 128, activation=tf.nn.relu)
            out = tf.layers.dense(out, 128, activation=tf.nn.relu)
            return out

    def __init__(self, state):
        state_shape = tf.shape(state)
        with tf.variable_scope('model'):
            def do_g_sum(state):
                # TODO: write test for this (maybe where relation is replaced with something simpler)
                # TODO: try making this part parallel too and see if it makes it faster
                # TODO: see if transposing and doing the map_fn over the objects would be faster.  i bet it might be
                x = tf.range(tf.shape(state)[0], dtype=tf.int32)
                idxs = cartesian_product(x,x)

                ijs = tf.reshape(tf.gather(state, idxs), [-1,2,2])
                ijs = tf.concat([ijs[:,0], ijs[:,1]], axis=1)
                g = self.relation(ijs)
                #return g
                return tf.reduce_sum(g, axis=0)

            #stateT = tf.transpose(state, [1,0,2])
            #g = tf.map_fn(do_g_sum, state, dtype=tf.float32)
            #g_sum = tf.reduce_sum(g, axis=1)
            g_sum = tf.map_fn(do_g_sum, state, dtype=tf.float32)
            self.f_out = self.f(g_sum)

            with tf.variable_scope('mdn'):
                self.locs = tf.reshape(tf.layers.dense(self.f_out, 2*FLAGS['k'], activation=None), [-1,FLAGS['k'],2])
                self.scales = tf.reshape(tf.layers.dense(self.f_out, 2*FLAGS['k'], activation=tf.exp), [-1,FLAGS['k'],2])
                self.logits = tf.layers.dense(self.f_out, FLAGS['k'], activation=None)

                cat = tfd.Categorical(logits=self.logits)
                # TODO: does this need to be a more complex normal
                components = []
                eval_components = []
                for loc, scale in zip(tf.unstack(tf.transpose(self.locs, [1,0,2])), tf.unstack(tf.transpose(self.scales, [1,0,2]))):
                    normal = tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale)
                    components.append(normal)
                    eval_normal = tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale)
                    eval_components.append(eval_normal)

                self.mixture = tfd.Mixture(cat=cat, components=components)
                self.eval_mixture = tfd.Mixture(cat=cat, components=eval_components)
            #self.mixture = tfd.Independent(self.mixture, reinterpreted_batch_dims=1)

            num_objs = tf.shape(state)[-2]

            tstate = tf.transpose(state, [1,0,2])

            loss = -tf.map_fn(self.mixture.log_prob, tstate, tf.float32)
            self.loss = tf.reduce_mean(loss)
            self.train_op = tf.train.AdamOptimizer(learning_rate=FLAGS['lr']).minimize(self.loss)

            with tf.variable_scope('eval'):
                self.X, self.Y = tf.meshgrid(tf.linspace(-1.0,1.0,100), tf.linspace(-1.0,1.0,100))
                self.stacked = tf.stack([self.X,self.Y], axis=-1)[:,:,None,:]
                self.evalZ = self.eval_mixture.log_prob(self.stacked)
                self.samples = self.eval_mixture.sample([1000])

            self.pred_plot_ph = tf.placeholder(tf.string)
            pred_plot = tf.image.decode_png(self.pred_plot_ph, channels=4)
            pred_plot = tf.expand_dims(pred_plot, 0)

            plot_summaries = []
            plot_summaries.append(tf.summary.image('mdn_contour', pred_plot))
            self.plot_summaries = tf.summary.merge(plot_summaries)

            eval_summaries = []
            with tf.name_scope('train'):
                eval_summaries.append(tf.summary.scalar('loss', self.loss))
                eval_summaries.append(tf.summary.scalar('min_logits', tf.reduce_min(self.logits[0])))
                eval_summaries.append(tf.summary.scalar('max_logits', tf.reduce_max(self.logits[0])))
                eval_summaries.append(tf.summary.scalar('median_logits', tfd.percentile(self.logits[0], 50.0)))

            self.summary = tf.summary.merge(eval_summaries)
            self.train_vals = {'loss': self.loss, 'train_op': self.train_op}
            self.eval_vals = {'state': state, 'samples': self.samples, 'summary': self.summary, 'loss': self.loss, 'X': self.X, 'Y': self.Y, 'Z': self.evalZ,
            'logits': self.logits}

def main():
    train_ds = Dataset(FLAGS)
    model = RNModel(train_ds.state['state'])
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter(FLAGS['log_path'] + '/train', sess.graph)

    try:
        pbar = tqdm.tqdm(total=1000)
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
            if i % 1000 == 0:
                pbar.close()
                pbar = tqdm.tqdm(total=1000)

            if i >= 5e4:
                exit()

    except KeyboardInterrupt:
        import ipdb; ipdb.set_trace()


# inverse of tf.nn.softplus
softplus_inverse = lambda x: tf.log(tf.math.expm1(x))

# TODO: strip out components, and just try AE

class ConvVAE(object):
    """
    Copied from:
    https://github.com/tensorflow/probability/blob/master/tensorflow_probability/examples/vae.py
    https://github.com/hardmaru/WorldModelsExperiments/blob/master/carracing/vae/vae.py
    """
    activation = tf.nn.relu
    #activation = tf.nn.leaky_relu
    def encoder(self, images):
        with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
            h = images
            h = tf.layers.conv2d(h, 32,  4, strides=2, activation=ConvVAE.activation, name="conv1")
            h = tf.layers.conv2d(h, 64,  4, strides=2, activation=ConvVAE.activation, name="conv2")
            h = tf.layers.conv2d(h, 128, 4, strides=2, activation=ConvVAE.activation, name="conv3")
            h = tf.layers.conv2d(h, 256, 4, strides=2, activation=ConvVAE.activation, name="conv4")
            h = tf.layers.flatten(h)

            loc = tf.layers.dense(h, FLAGS['z_size'], activation=None, name='fc_mu')
            log_scale = tf.layers.dense(h, FLAGS['z_size'], activation=None, name='fc_log_var')
            scale = tf.nn.softplus(log_scale + softplus_inverse(1.0)) # idk what this is for. maybe ensuring center around 1.0

            return tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale, name='code')

    def decoder(self, codes):
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            original_shape = tf.shape(codes)
            h = codes
            h = tf.layers.dense(h, 4*256, name='fc')
            h = tf.reshape(h, [-1, 1, 1, 4*256])
            h = tf.layers.conv2d_transpose(h, 128, 5, strides=2, activation=ConvVAE.activation, name="deconv1")
            h = tf.layers.conv2d_transpose(h, 64,  5, strides=2, activation=ConvVAE.activation, name="deconv2")
            h = tf.layers.conv2d_transpose(h, 32,  6, strides=2, activation=ConvVAE.activation, name="deconv3")
            logits = tf.layers.conv2d_transpose(h, 1, 6, strides=2, activation=None, name="deconv4")
            logits = tf.reshape(logits, shape=tf.concat([original_shape[:-1], IMAGE_SHAPE], axis=0))

            return tfd.Independent(tfd.Bernoulli(logits=logits), reinterpreted_batch_ndims=len(IMAGE_SHAPE), name="image") 

    def mixture_prior(self):
        """Unsure exactly how this works.  I thought it was supposed to use a single Gaussian prior"""
        loc = tf.get_variable(name="loc", shape=[FLAGS['vae_k'], FLAGS['z_size']])
        raw_scale_diag = tf.get_variable(name="raw_scale_diag", shape=[FLAGS['vae_k'], FLAGS['z_size']])
        mixture_logits = tf.get_variable(name="mixture_logits", shape=[FLAGS['vae_k']])
        
        return tfd.MixtureSameFamily(
            components_distribution=tfd.MultivariateNormalDiag(
                loc=loc, 
                scale_diag=tf.nn.softplus(raw_scale_diag)), 
                mixture_distribution=tfd.Categorical(logits=mixture_logits), 
            name="prior")

    def __init__(self, state):
        self.state = state
        images = self.state['image']

        self.run_vals = {}
        latent_prior = self.mixture_prior()

        approx_posterior = self.encoder(images)
        approx_posterior_sample = approx_posterior.sample(FLAGS['num_vae_samples'])
        decoder_likelihood = self.decoder(approx_posterior_sample)

        self.samples = decoder_likelihood.sample()

        distortion = -decoder_likelihood.log_prob(images)
        self.mean_distortion = tf.reduce_mean(distortion)

        # approximate kl divergence trying to match approx_posterior to latent_prior (I think)
        rate = (approx_posterior.log_prob(approx_posterior_sample) - latent_prior.log_prob(approx_posterior_sample))
        self.mean_rate = tf.reduce_mean(rate)

        elbo_local = -(rate + distortion)

        self.elbo = tf.reduce_mean(elbo_local)
        self.loss = -self.elbo
        self.train_op = tf.train.AdamOptimizer(FLAGS['lr']).minimize(self.loss)

        # define summaries
        with tf.name_scope('train'):
            image_tile_summary('input', tf.to_float(images), rows=1, cols=8)

            image_tile_summary("recon/sample", tf.to_float(decoder_likelihood.sample()[:1, :8]), rows=1, cols=8)
            recon_mean = decoder_likelihood.mean()[:1, :8]
            recon_clean = tf.math.round(recon_mean)
            image_tile_summary("recon/mean", recon_mean, rows=1, cols=8)
            image_tile_summary("recon/clean", recon_clean, rows=1, cols=8)

            # Decode samples from the prior for visualization.
            random_image = self.decoder(latent_prior.sample(16))
            image_tile_summary('random/sample', tf.to_float(random_image.sample()), rows=4, cols=4)
            image_tile_summary('random/mean', random_image.mean(), rows=4, cols=4)


        self.summary = tf.summary.merge_all(scope='train')
        self.train_vals = {'loss': self.loss, 'train_op': self.train_op}
        self.eval_vals = {'samples': self.samples, 'state': self.state, 'summary': self.summary, 'loss': self.loss}



def vae_main():
    train_ds = Dataset(FLAGS)
    model = ConvVAE(train_ds.state)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    train_writer = tf.summary.FileWriter(FLAGS['log_path'] + '/train', sess.graph)

    pbar = tqdm.tqdm(total=100)
    for i in itertools.count(start=1):
        train_vals = sess.run(model.train_vals)
        pbar.update()

        if i % 100 == 0:
            pbar.close()
            pbar = tqdm.tqdm(total=100)
            eval_vals = sess.run(model.eval_vals)
            train_writer.add_summary(eval_vals['summary'], global_step=i)
            #plot_arr(curr_state['image'][0][...,0])
            #plot_in_out_vae(curr_state['image'][0][...,0], samples[0][0][...,0], FLAGS, i=i)
            #print('scales: min: {} max: {} mean: {} median: {}'.format(np.min(scales[0]), np.max(scales[0]), np.mean(scales[0]), np.median(scales[0])))
            #print('i = {}, loss = {}'.format(i, eval_vals['loss']))

        if i >= 5e4:
            exit()

if __name__ == "__main__":
    if FLAGS['plot_shapes']:
        #while True:
        #    plot_shapes(data_generator(FLAGS['num_shapes']), FLAGS)
        while True:
            dg = data_generator(FLAGS['num_shapes'], FLAGS['samplers'])
            plot_arr(dg.__next__()['image'][...,0])

    elif FLAGS['vae']:
        vae_main()
    else:
        main()
