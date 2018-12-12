import tensorflow as tf
from tensorflow_probability import distributions as tfd
from rns.building_blocks import f_net, relation_sum, relation_sum_transpose, encoder_conv, encoder_net, decoder_net, mdn_head, mixture_prior, snub_encoder, softplus_inverse, relation_net
from rns.util import image_tile_summary, cartesian_product
from rns.constant import W, H, R, IMAGE_SHAPE
from rns.viz import plot

class Model(object):
    def __init__(self, state, FLAGS, name='model'):
        self.FLAGS = FLAGS
        self.name = name
        self.state = state
        self.scalar_summaries = {}
        self.img_summaries = {}
        self.train_vals = {}
        self.eval_vals = {}

    def forward(self, inputs):
        raise NotImplementedError

class Autoencoder(Model):
    def encoder(self, images):
        raise NotImplementedError
    def decoder(self, codes):
        raise NotImplementedError

class RNModel(Model):
    """Relation network model that takes in raw state"""
    def forward(self, inputs):
        with tf.variable_scope(self.name):
            g_sum = relation_sum(inputs, self.FLAGS)
            f_out = f_net(g_sum, self.FLAGS)
            mdn = mdn_head(f_out, self.FLAGS)
        return mdn

    def __init__(self, state, FLAGS, name='RN'):
        super().__init__(state, FLAGS, name)
        self.state = self.state['state']
        self.mdn = self.forward(self.state)

        # calculate log_probability over all objects in the set for loss
        tstate = tf.transpose(self.state, [1,0,2])
        self.loss = -tf.map_fn(self.mdn['mixture'].log_prob, tstate, tf.float32)
        self.loss = tf.reduce_mean(self.loss)
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.FLAGS['lr']).minimize(self.loss)

        # generate logging info
        with tf.variable_scope('eval'):
            self.X, self.Y = tf.meshgrid(tf.linspace(-1.0,1.0,100), tf.linspace(-1.0,1.0,100))
            self.stacked = tf.stack([self.X,self.Y], axis=-1)[:,:,None,:]
            self.evalZ = self.mdn['eval_mixture'].log_prob(self.stacked)
            self.samples = self.mdn['eval_mixture'].sample([1000])

        self.pred_plot_ph = tf.placeholder(tf.string)
        pred_plot = tf.image.decode_png(self.pred_plot_ph, channels=4)
        pred_plot = tf.expand_dims(pred_plot, 0)

        plot_summaries = []
        plot_summaries.append(tf.summary.image('mdn_contour', pred_plot))
        self.plot_summaries = tf.summary.merge(plot_summaries)

        eval_summaries = []
        with tf.name_scope('train'):
            eval_summaries.append(tf.summary.scalar('loss', self.loss))
            eval_summaries.append(tf.summary.scalar('min_logits', tf.reduce_min(self.mdn['logits'][0])))
            eval_summaries.append(tf.summary.scalar('max_logits', tf.reduce_max(self.mdn['logits'][0])))
            eval_summaries.append(tf.summary.scalar('median_logits', tfd.percentile(self.mdn['logits'], 50.0)))

        # bundle up 
        self.summary = tf.summary.merge(eval_summaries)
        self.train_vals = {'loss': self.loss, 'train_op': self.train_op}
        #self.eval_vals = {'state': self.state, 'summary': self.summary, 'loss': self.loss, 'X': self.X, 'Y': self.Y, 'Z': self.evalZ, 'logits': self.mdn['logits']}
        self.eval_vals = {'state': self.state, 'samples': self.samples, 'summary': self.summary, 'loss': self.loss, 'X': self.X, 'Y': self.Y, 'Z': self.evalZ, 'logits': self.mdn['logits']}

class SingletonModel(RNModel):
    """Singleton network model that takes in raw state"""
    def forward(self, inputs):
        with tf.variable_scope(self.name):
            def do_g_sum(objs):
                g = relation_net(objs, self.FLAGS)
                return tf.reduce_sum(g, axis=0)
            
            g_sum = tf.map_fn(do_g_sum, inputs, dtype=tf.float32)

            f_out = f_net(g_sum, self.FLAGS)
            mdn = mdn_head(f_out, self.FLAGS)
        return mdn

    def __init__(self, state, FLAGS, name='Singleton'):
        super().__init__(state, FLAGS, name)


class ConvAE(Autoencoder):
    """Standard Autoencoder with MSE loss"""
    def encoder(self, images):
        with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
            enc_out = encoder_net(images, self.FLAGS, scope=None)
            return enc_out['loc']
    def decoder(self, codes):
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            logits = decoder_net(codes, scope=None)
            return logits
            #logits = decoder_net(codes, scope=None)
            #return tfd.Independent(tfd.Bernoulli(logits=logits), reinterpreted_batch_ndims=len(IMAGE_SHAPE), name="image") 

    def forward(self, inputs):
        with tf.variable_scope(self.name):
            codes = self.encoder(inputs)
            decoded = self.decoder(codes)
            return decoded

    def __init__(self, state, FLAGS, name='ConvAE'):
        super().__init__(state, FLAGS, name)
        images = self.state['image']
        outputs = self.forward(images)

        #self.loss = tf.reduce_mean(-outputs.log_prob(images))
        self.loss = tf.losses.mean_squared_error(images, outputs)
        #self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=images, logits=outputs))
        self.train_op = tf.train.AdamOptimizer(self.FLAGS['lr']).minimize(self.loss)

        # define summaries
        with tf.name_scope('train'):
            tf.summary.scalar('loss', self.loss)
            image_tile_summary('inputs', tf.to_float(images), rows=1, cols=8)
            image_tile_summary('outputs', tf.to_float(outputs), rows=1, cols=8)
            image_tile_summary('outputs_clean', tf.round(tf.to_float(outputs)), rows=1, cols=8)
            #recon_mean = outputs.mean()[:1, :8]
            #recon_clean = tf.math.round(recon_mean)
            #image_tile_summary('recon/mean', recon_mean, rows=1, cols=8)
            #image_tile_summary('recon/clean', recon_clean, rows=1, cols=8)


        self.summary = tf.summary.merge_all(scope='train')
        self.train_vals.update({'loss': self.loss, 'train_op': self.train_op})
        self.eval_vals.update({'state': self.state, 'summary': self.summary, 'loss': self.loss})


class ConvVAE(Autoencoder):
    """
    Variational Autoencoder 

    Copied from:
    https://github.com/tensorflow/probability/blob/master/tensorflow_probability/examples/vae.py
    https://github.com/hardmaru/WorldModelsExperiments/blob/master/carracing/vae/vae.py
    """
    activation = tf.nn.relu
    #activation = tf.nn.leaky_relu
    def encoder(self, images):
        with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
            enc_out = encoder_net(images, self.FLAGS, scope=None)
            return tfd.MultivariateNormalDiag(loc=enc_out['loc'], scale_diag=enc_out['scale'], name='code')

    def decoder(self, codes):
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            logits = decoder_net(codes, scope=None)
            return tfd.Independent(tfd.Bernoulli(logits=logits), reinterpreted_batch_ndims=len(IMAGE_SHAPE), name="image") 

    def forward(self, inputs):
        with tf.variable_scope(self.name):
            q = self.encoder(inputs)  # approximate posterior
            q_sample = q.sample(self.FLAGS['num_vae_samples'])  # approximate posterior sample
            p = self.decoder(q_sample)  # deoder likelihood
            return dict(q=q, q_sample=q_sample, p=p)

    def __init__(self, state, FLAGS, name='ConvVAE'):
        super().__init__(state, FLAGS, name)
        images = self.state['image']

        vae_vals = self.forward(images)

        # TODO: could maybe make these components reusable. VAE loss and summary

        distortion = -vae_vals['p'].log_prob(images)
        # approximate kl divergence trying to match approx_posterior to latent_prior (I think)
        latent_prior = mixture_prior(self.FLAGS)
        rate = (vae_vals['q'].log_prob(vae_vals['q_sample']) - latent_prior.log_prob(vae_vals['q_sample']))
        elbo_local = -(rate + distortion)

        self.elbo = tf.reduce_mean(elbo_local)
        self.loss = -self.elbo
        self.train_op = tf.train.AdamOptimizer(self.FLAGS['lr']).minimize(self.loss)

        # define summaries
        with tf.name_scope('train'):
            tf.summary.scalar('mean_distortion', tf.reduce_mean(distortion))
            tf.summary.scalar('mean_rate', tf.reduce_mean(rate))
            tf.summary.scalar('loss', self.loss)
            image_tile_summary('input', tf.to_float(images), rows=1, cols=8)
            image_tile_summary("recon/sample", tf.to_float(vae_vals['p'].sample()[:1, :8]), rows=1, cols=8)
            recon_mean = vae_vals['p'].mean()[:1, :8]
            recon_clean = tf.math.round(recon_mean)
            image_tile_summary('recon/mean', recon_mean, rows=1, cols=8)
            image_tile_summary('recon/clean', recon_clean, rows=1, cols=8)
            # Sample and decode from prior for visualization.
            samples = vae_vals['p'].sample()
            random_image = self.decoder(latent_prior.sample(16))
            image_tile_summary('random/sample', tf.to_float(random_image.sample()), rows=4, cols=4)
            image_tile_summary('random/mean', random_image.mean(), rows=4, cols=4)

        self.summary = tf.summary.merge_all(scope='train')
        self.train_vals.update({'loss': self.loss, 'train_op': self.train_op})
        self.eval_vals.update({'samples': samples, 'state': self.state, 'summary': self.summary, 'loss': self.loss})


# TODO: also try a version of this where it predicts xy coord distribution
class ConvRN_VAE(ConvVAE):
    def concat_coord(self, o, i, d):
        coor = tf.tile(tf.expand_dims([float(int(i / d)) / d, (i % d) / d], axis=0), [self.FLAGS['bs'], 1])
        o = tf.concat([o, tf.to_float(coor)], axis=1)
        return o

    def encoder(self, images):
        with tf.variable_scope('encoder'):
            h = images
            h = tf.layers.conv2d(h, 32, 3, strides=2, activation=tf.nn.relu, name='conv1')
            h = tf.layers.conv2d(h, 64, 3, strides=2, activation=tf.nn.relu, name='conv2')
            h = tf.layers.conv2d(h, 128, 3, strides=2, activation=tf.nn.relu, name='conv3')
            h = tf.layers.conv2d(h, 32, 3, strides=2, activation=tf.nn.relu, name='conv4')
            d = h.get_shape().as_list()[1]
            all_g = []
            for i in range(d*d):
                o_i = h[:, int(i / d), int(i % d), :]
                o_i = self.concat_coord(o_i, i, d)
                for j in range(d*d):
                    o_j = h[:, int(j / d), int(j % d), :]
                    o_j = self.concat_coord(o_j, j, d)
                    oij = tf.concat([o_i, o_j], axis=1)
                    g_i_j = relation_net(oij, self.FLAGS)
                    all_g.append(g_i_j)
            all_g = tf.stack(all_g, axis=0)
            all_g = tf.reduce_mean(all_g, axis=0, name='all_g')

            #conv_shape = tf.shape(h)
            #new_shape = tf.stack([conv_shape[0], -1, 256], axis=0)
            #conv_objs = tf.reshape(h, new_shape)
            #with tf.device('/cpu:0'):
            #    g_sum = relation_sum(conv_objs, self.FLAGS)
            f_out = f_net(all_g, FLAGS)
            loc = tf.layers.dense(f_out, self.FLAGS['z_size'], activation=None, name='fc_mu')
            log_scale = tf.layers.dense(f_out, self.FLAGS['z_size'], activation=None, name='fc_log_var')
            scale = tf.nn.softplus(log_scale + softplus_inverse(1.0)) # idk what this is for. maybe ensuring center around 1.0
            return tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale, name='code')

    def decoder(self, codes):
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            logits = decoder_net(codes, scope=None)
            return tfd.Independent(tfd.Bernoulli(logits=logits), reinterpreted_batch_ndims=len(IMAGE_SHAPE), name="image") 

    def forward(self, inputs):
        with tf.variable_scope(self.name):
            q = self.encoder(inputs)  # approximate posterior
            q_sample = q.sample(self.FLAGS['num_vae_samples'])  # approximate posterior sample
            p = self.decoder(q_sample)  # deoder likelihood
            return dict(q=q, q_sample=q_sample, p=p)

    def __init__(self, state, FLAGS, name='ConvVAE'):
        super().__init__(state, FLAGS, name)


class State2ImageVAE(ConvVAE):
    def encoder(self, inputs):
        with tf.variable_scope('encoder'):
            #with tf.device('/cpu:0'):
            g_sum = relation_sum(inputs, self.FLAGS)
            f_out = f_net(g_sum, FLAGS)
            loc = tf.layers.dense(f_out, self.FLAGS['z_size'], activation=None, name='fc_mu')
            log_scale = tf.layers.dense(f_out, self.FLAGS['z_size'], activation=None, name='fc_log_var')
            scale = tf.nn.softplus(log_scale + softplus_inverse(1.0)) # idk what this is for. maybe ensuring center around 1.0
            return tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale, name='code')

class ConvMDN(Model):
    def forward(self, inputs):
        with tf.variable_scope(self.name):
            h = encoder_conv(inputs)
            h = tf.layers.flatten(h)
            #h = tf.contrib.layers.spatial_softmax(h)
            mdn = mdn_head(h, self.FLAGS)
        return mdn

    def __init__(self, state, FLAGS, name='ConvMDN'):
        super().__init__(state, FLAGS, name)
        self.image = self.state['image']
        self.state = self.state['state']
        self.mdn = self.forward(self.image)

        # calculate log_probability over all objects in the set for loss
        tstate = tf.transpose(self.state, [1,0,2])
        self.loss = -tf.map_fn(self.mdn['mixture'].log_prob, tstate, tf.float32)
        self.loss = tf.reduce_mean(self.loss)
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.FLAGS['lr']).minimize(self.loss)

        # generate logging info
        with tf.variable_scope('eval'):
            self.X, self.Y = tf.meshgrid(tf.linspace(-1.0,1.0,100), tf.linspace(-1.0,1.0,100))
            self.stacked = tf.stack([self.X,self.Y], axis=-1)[:,:,None,:]
            self.evalZ = self.mdn['eval_mixture'].log_prob(self.stacked)
            self.samples = self.mdn['eval_mixture'].sample([1000])

        self.pred_plot_ph = tf.placeholder(tf.string)
        pred_plot = tf.image.decode_png(self.pred_plot_ph, channels=4)
        pred_plot = tf.expand_dims(pred_plot, 0)

        plot_summaries = []
        plot_summaries.append(tf.summary.image('mdn_contour', pred_plot))
        self.plot_summaries = tf.summary.merge(plot_summaries)

        eval_summaries = []
        with tf.name_scope('train'):
            eval_summaries.append(tf.summary.scalar('loss', self.loss))
            eval_summaries.append(tf.summary.scalar('min_logits', tf.reduce_min(self.mdn['logits'][0])))
            eval_summaries.append(tf.summary.scalar('max_logits', tf.reduce_max(self.mdn['logits'][0])))
            eval_summaries.append(tf.summary.scalar('median_logits', tfd.percentile(self.mdn['logits'], 50.0)))

        # bundle up 
        self.summary = tf.summary.merge(eval_summaries)
        self.train_vals = {'loss': self.loss, 'train_op': self.train_op}
        self.eval_vals = {'state': self.state, 'samples': self.samples, 
        'summary': self.summary, 'loss': self.loss, 'X': self.X, 'Y': self.Y, 'Z': self.evalZ, 'logits': self.mdn['logits']}