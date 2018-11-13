import os
import itertools
import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from define_flags import FLAGS
from rns.data import normalize, subsample_postbatch, to_float, data_generator, Dataset
from rns.viz import plot_contour, plot_samples, plot_shapes, plot_arr


def relation(oij, scope='g'):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE) as scope:
        #out = tf.concat([objects[ij[0]],objects[ij[0]]])
        out = oij
        out = tf.layers.dense(out, 128, activation=tf.nn.relu)
        out = tf.layers.dense(out, 128, activation=tf.nn.relu)
        out = tf.layers.dense(out, 128, activation=None)
    return out

def f(g_sum):
    out = g_sum
    out = tf.layers.dense(out, 128, activation=tf.nn.relu)
    out = tf.layers.dense(out, 128, activation=tf.nn.relu)
    out = tf.layers.dense(out, 128, activation=tf.nn.relu)
    return out

def cartesian_product(a,b):
    a, b = a[None, :, None], b[:, None, None]
    prod = tf.concat([b + tf.zeros_like(a), tf.zeros_like(b) + a], axis = 2)
    #new_shape = tf.stack([-1, tf.shape(cartesian_product)[-1]])
    #cartesian_product = tf.reshape(cartesian_product, new_shape)
    prod = tf.reshape(prod, [-1])
    return prod

class RNModel(object):
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
                g = relation(ijs)
                return tf.reduce_sum(g, axis=0)

            g_sum = tf.map_fn(do_g_sum, state, dtype=tf.float32)
            self.f_out = f(g_sum)

            self.locs = tf.reshape(tf.layers.dense(self.f_out, 2*FLAGS['k'], activation=None), [-1,FLAGS['k'],2])
            self.scales = tf.reshape(tf.layers.dense(self.f_out, 2*FLAGS['k'], activation=tf.exp), [-1,FLAGS['k'],2])
            self.logits = tf.layers.dense(self.f_out, FLAGS['k'], activation=None)

            cat = tfd.Categorical(logits=self.logits)
            # TODO: does this need to be a more complex normal
            components = []
            eval_components = []
            for loc, scale in zip(tf.unstack(tf.transpose(self.locs, [1,0,2])), tf.unstack(tf.transpose(self.scales, [1,0,2]))):
                # tile these so that each of the samples share the same distribution values (they are iid)
                #tiled_loc = tf.tile(loc[:,None,:], [1,state_shape[-2],1])
                #tiled_scale = tf.tile(scale[:,None,:], [1,state_shape[-2],1])
                #normal = tfd.MultivariateNormalDiag(loc=tiled_loc, scale_diag=tiled_scale)
                normal = tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale)
                # collapse the distro down so that they are treated in an event_shape
                #dist = tfd.Independent(normal, reinterpreted_batch_ndims=1) 
                #components.append(dist)
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

            #loss = 0
            #for i in range(10):
            #    loss += -self.mixture.log_prob(state[:,i])
            self.loss = tf.reduce_mean(loss)
            self.train_op = tf.train.AdamOptimizer(learning_rate=FLAGS['lr']).minimize(self.loss)

            self.X, self.Y = tf.meshgrid(tf.linspace(-1.0,1.0,100), tf.linspace(-1.0,1.0,100))
            self.stacked = tf.stack([self.X,self.Y], axis=-1)[:,:,None,:]
            self.eval = self.eval_mixture.log_prob(self.stacked)
            self.samples = self.eval_mixture.sample([1000])


    


def main():
    train_ds = Dataset(FLAGS)
    model = RNModel(train_ds.state)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    try:
        for i in itertools.count(start=1):
            _, loss = sess.run([model.train_op, model.loss])

            if i % 100 == 0:
                samples, logits, locs, scales, curr_state, loss, X, Y, Z = sess.run([model.samples, model.logits, model.locs, model.scales, train_ds.state, model.loss, model.X, model.Y, model.eval])

                plot_contour(curr_state, X, Y, Z, FLAGS, i=i)

                print(scales[0])
                print(logits[0])
                print('i = {}, loss = {}'.format(i, loss))

            if i >= 5e4:
                exit()

    except KeyboardInterrupt:
        import ipdb; ipdb.set_trace()


if __name__ == "__main__":
    if FLAGS['plot_shapes']:
        #while True:
        #    plot_shapes(data_generator(FLAGS['num_shapes']), FLAGS)
        while True:
            dg = data_generator(FLAGS['num_shapes'])
            plot_arr(dg.__next__()['image'])
    else:
        main()
