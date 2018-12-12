import tensorflow as tf
from tensorflow_probability import distributions as tfd
from rns.constant import W, H, R, IMAGE_SHAPE
from rns.util import cartesian_product

"""
LEGO bricks
"""

def relation_net(oij, FLAGS, scope='g'):
    """MLP for relation (g)  part of Relation Network (RN)"""
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        h = oij
        #hs = []
        for i in range(FLAGS['num_g_fc']-1):
            h = tf.layers.dense(h, FLAGS['g_size'], activation=tf.nn.relu)
            #hs.append(h)
        h = tf.layers.dense(h, FLAGS['gf_x'], activation=tf.nn.relu)
    return h 

def _f_net(g_sum, FLAGS, scope='f'):
    """MLP for f part of RN"""
    with tf.variable_scope(scope):
        h = g_sum
        #hs = []
        h = tf.layers.dense(h, FLAGS['gf_x'], activation=tf.nn.relu)
        for i in range(FLAGS['num_f_fc']-1):
            h = tf.layers.dense(h, FLAGS['f_size'], activation=tf.nn.relu)
            #hs.append(h)
        return h


def f_net(g_sum, FLAGS, scope='f'):
    if FLAGS['skip']:
        return _skip_f_net(g_sum, FLAGS, scope)
    else:
        return _f_net(g_sum, FLAGS, scope)


# TODO: wonder if it would help to divide by sqrt of number of objects in order to make magnitude more similar
def relation_sum(objs_batch, FLAGS):
    """Take in 'objects' as list. Paralleize over batch""" 
    # TODO: write test for this (maybe where relation is replaced with something simpler)
    # TODO: try making this part parallel too and see if it makes it faster
    def do_g_sum(objs):
        x = tf.range(tf.shape(objs)[0], dtype=tf.int32)
        idxs = cartesian_product(x,x)

        ijs = tf.reshape(tf.gather(objs, idxs), [-1,2,2])
        ijs = tf.concat([ijs[:,0], ijs[:,1]], axis=1)
        if FLAGS['skip']:
            g = skip_relation_net(ijs, FLAGS)
        else:
            g = relation_net(ijs, FLAGS)
        return tf.reduce_sum(g, axis=0)

    g_sum = tf.map_fn(do_g_sum, objs_batch, dtype=tf.float32)
    return g_sum

# TODO: do a comparison between these to see which one is faster
def relation_sum_transpose(objs, FLAGS):
    """
    Take in 'objects' as list. Parallelize (map_fn) over object pairs instead of batch

    I am not totally sure this will produce the same results as the other version
    """ 
    raise NotImplementedError('Not Implemented Correctly Exception')

    def do_g(batch_ij):
        x = tf.range(tf.shape(batch_ij)[0], dtype=tf.int32)
        idxs = cartesian_product(x,x)
        ijs = tf.reshape(tf.gather(batch_ij, idxs), [-1,2,2])
        ijs = tf.concat([ijs[:,0], ijs[:,1]], axis=1)
        g = relation_net(ijs, FLAGS)
        return g

    # Change order of dims to do map_fn over the objects.  This might be faster if there are many more object pairs than batches. Or maybe the opposite of that
    #import ipdb; ipdb.set_trace()
    objsT = tf.transpose(objs, [1,0,2])
    g = tf.map_fn(do_g, objsT, dtype=tf.float32)
    g_sum = tf.reduce_sum(g, axis=1)
    return g_sum

def mdn_head(h, FLAGS):
    with tf.variable_scope('mdn'):
        locs = tf.reshape(tf.layers.dense(h, 2*FLAGS['k'], activation=None), [-1,FLAGS['k'],2])
        scales = tf.reshape(tf.layers.dense(h, 2*FLAGS['k'], activation=tf.exp), [-1,FLAGS['k'],2])
        logits = tf.layers.dense(h, FLAGS['k'], activation=None)

        cat = tfd.Categorical(logits=logits)
        components = []
        eval_components = []
        for loc, scale in zip(tf.unstack(tf.transpose(locs, [1,0,2])), tf.unstack(tf.transpose(scales, [1,0,2]))):
            # TODO: does this need to be a more complex distribution?
            normal = tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale)
            components.append(normal)
            eval_normal = tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale)
            eval_components.append(eval_normal)
        mixture = tfd.Mixture(cat=cat, components=components)
        eval_mixture = tfd.Mixture(cat=cat, components=eval_components)
    return {'locs': locs, 'scales': scales, 'logits': logits, 'mixture': mixture, 'eval_mixture': eval_mixture}

# Conv and Auto-Encoder stuff
# inverse of tf.nn.softplus
softplus_inverse = lambda x: tf.log(tf.math.expm1(x))

def snub_encoder(images, activation=tf.nn.relu):
    h = images
    h = tf.layers.conv2d(h, 32,  4, strides=2, activation=activation, name="conv1")
    h = tf.layers.conv2d(h, 64,  4, strides=2, activation=activation, name="conv2")
    return h

def encoder_conv(images, activation=tf.nn.relu):
    h = images
    h = tf.layers.conv2d(h, 32,  4, strides=2, activation=activation, name="conv1")
    h = tf.layers.conv2d(h, 64,  4, strides=2, activation=activation, name="conv2")
    h = tf.layers.conv2d(h, 128, 4, strides=2, activation=activation, name="conv3")
    h = tf.layers.conv2d(h, 256, 4, strides=2, activation=activation, name="conv4")
    return h

def encoder_net(images, FLAGS, scope='encoder', reuse=tf.AUTO_REUSE, activation=tf.nn.relu):
    # TODO: probably move to other function
    h = encoder_conv(images, activation=activation)
    h = tf.layers.flatten(h)
    loc = tf.layers.dense(h, FLAGS['z_size'], activation=None, name='fc_mu')
    log_scale = tf.layers.dense(h, FLAGS['z_size'], activation=None, name='fc_log_var')
    scale = tf.nn.softplus(log_scale + softplus_inverse(1.0)) # idk what this is for. maybe ensuring center around 1.0
    return {'loc': loc, 'scale': scale}

def decoder_net(codes, scope='decoder', reuse=tf.AUTO_REUSE, activation=tf.nn.relu):
    original_shape = tf.shape(codes)
    h = codes
    h = tf.layers.dense(h, 4*256, name='fc')
    h = tf.reshape(h, [-1, 1, 1, 4*256])
    h = tf.layers.conv2d_transpose(h, 128, 5, strides=2, activation=activation, name="deconv1")
    h = tf.layers.conv2d_transpose(h, 64,  5, strides=2, activation=activation, name="deconv2")
    h = tf.layers.conv2d_transpose(h, 32,  6, strides=2, activation=activation, name="deconv3")
    logits = tf.layers.conv2d_transpose(h, 1, 6, strides=2, activation=None, name="deconv4")
    logits = tf.reshape(logits, shape=tf.concat([original_shape[:-1], IMAGE_SHAPE], axis=0))
    return logits

def mixture_prior(FLAGS):
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


def skip_relation_net(oij, scope='g'):
    """MLP for relation (g)  part of Relation Network (RN)"""
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        #out = tf.concat([objects[ij[0]],objects[ij[0]]])
        h = oij
        h1 = tf.layers.dense(h, 128, activation=tf.nn.relu)
        h2 = tf.layers.dense(h1, 128, activation=tf.nn.relu)
        h2 = tf.contrib.layers.layer_norm(h1 + h2)
        h3 = tf.layers.dense(h2, 128, activation=tf.nn.relu)
        h3 = tf.contrib.layers.layer_norm(h3 + h2)
        h4 = tf.layers.dense(h3, 128, activation=None)
        h4 = tf.contrib.layers.layer_norm(h4 + h3)
    out = h4
    return out

def _skip_f_net(g_sum, scope='f'):
    """MLP for f part of RN"""
    with tf.variable_scope(scope):
        h = g_sum
        h1 = tf.layers.dense(h, 128, activation=tf.nn.relu)
        h2 = tf.layers.dense(h1, 128, activation=tf.nn.relu)
        h2 = tf.contrib.layers.layer_norm(h1 + h2)
        h3 = tf.layers.dense(h2, 128, activation=tf.nn.relu)
        h3 = tf.contrib.layers.layer_norm(h2 + h3)
        h4 = tf.layers.dense(h3, 128, activation=tf.nn.relu)
        h4 = tf.contrib.layers.layer_norm(h3 + h4)
        h5 = tf.layers.dense(h4, 1024, activation=tf.nn.relu)
    out = h5
    return out

