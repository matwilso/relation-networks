import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import itertools
import tensorflow as tf

from tensorflow_probability import distributions as tfd
from tensorflow_probability import edward2 as ed

BS = 30
N = 10
W = 64  
H = 64  
R = 2
K = 10
DIRS = list(itertools.product([-4,0,4], [-4,0,4]))
DIRS.remove((0,0))

class Circle(object):
    def __init__(self, x, y, r=R):
        self.x = x
        self.y = y
        self.r = r

    @staticmethod
    def sample(xlim, ylim=None):
        x = np.random.randint(xlim[0],xlim[1])
        if ylim is None:
            y = np.random.randint(xlim[0],xlim[1])
        else:
            y = np.random.randint(ylim[0],ylim[1])
        return Circle(x, y)

    def plot(self, ax):
        circ = mpatches.Circle((self.x,self.y), self.r, color='C1')
        ax.add_patch(circ)
    
    def __str__(self):
        return 'x: {} y: {} r: {}'.format(self.x, self.y, self.r)

    def __eq__(self, other):
        me = np.array([self.x, self.y])
        them = np.array([other.x, other.y])
        dist = np.linalg.norm(me-them) 
        return dist < 2*np.max([self.r, other.r])

    @staticmethod
    def list_to_state(objs):
        arr = []
        for obj in objs:
            arr.append([obj.x, obj.y])
        return np.array(arr)

def uniform(n):
    objs = []
    for i in range(n):
        while True:
            c = Circle.sample([R,W-R])
            if c in objs:
                continue
            else:
                break
        objs.append(c)
    return {'shapes': objs, 'state': objs[0].list_to_state(objs)}

def cluster1(n):
    border = 2
    objs = [Circle.sample([border,W-border])]

    for i in range(n-1):
        while True:
            reference = np.random.choice(objs)
            dir = np.array(DIRS[np.random.randint(len(DIRS))])
            dir *= np.random.randint(1,3)

            x, y = reference.x + dir[0], reference.y + dir[1]
            c = Circle(x,y)

            if (c in objs) or c.x > W-R or c.x < R or c.y > H-R or c.y < R:
                continue
            else:
                break


        objs.append(c)
    return {'shapes': objs, 'state': objs[0].list_to_state(objs)}

def cluster2(n):
    border = 2

    g1 = [Circle.sample([border,W-border])]
    while True:
        g2 = [Circle.sample([border,W-border])]
        if g1[0] != g2[0]:
            break

    for i in range(n-2):
        while True:
            g = g1 if np.random.binomial(1,0.5) else g2
            reference = np.random.choice(g)
            dir = np.array(DIRS[np.random.randint(len(DIRS))])
            dir *= np.random.randint(1,3)

            x, y = reference.x + dir[0], reference.y + dir[1]
            c = Circle(x,y)

            if (c in g1+g2) or c.x > W-2 or c.x < 2 or c.y > H-2 or c.y < 2:
                continue
            else:
                break
        g.append(c)

    objs = g1+g2
    return {'shapes': objs, 'state': objs[0].list_to_state(objs)}

def data_generator(n, key='state'):
    while True:
        #sampler = np.random.choice([uniform, cluster1, cluster2])
        sampler = uniform
        samples = sampler(n)
        yield samples[key]


def subsample_postbatch(state):
    backset = tf.cast(6*tf.random.uniform([]), tf.int32)
    return state[:,:N-backset]
    #return tf.gather(state, tf.range(idx), axis=1)

def to_float(state):
    return tf.to_float(state)

def normalize(state):
    return (state - W//2) / (W//2)


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

class Model(object):
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

            self.locs = tf.reshape(tf.layers.dense(self.f_out, 2*K, activation=None), [-1,K,2])
            self.scales = tf.reshape(tf.layers.dense(self.f_out, 2*K, activation=tf.exp), [-1,K,2])
            self.logits = tf.layers.dense(self.f_out, K, activation=None)

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
            self.train_op = tf.train.AdamOptimizer(learning_rate=3e-4).minimize(self.loss)

            self.X, self.Y = tf.meshgrid(tf.linspace(-1.0,1.0,100), tf.linspace(-1.0,1.0,100))
            self.stacked = tf.stack([self.X,self.Y], axis=-1)[:,:,None,:]
            self.eval = self.eval_mixture.log_prob(self.stacked)



def main():
    dg = lambda : data_generator(N)
    ds = tf.data.Dataset.from_generator(dg, tf.int64, tf.TensorShape([None,2]))
    ds = ds.map(to_float)
    ds = ds.batch(BS)
    ds = ds.map(normalize)
    #ds = ds.map(subsample_postbatch)
    ds = ds.prefetch(10)

    iterator = ds.make_one_shot_iterator()
    state = iterator.get_next()

    model = Model(state)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    try:
        for i in itertools.count(start=1):
            _, loss = sess.run([model.train_op, model.loss])

            if i % 100 == 0:
                logits, locs, scales, curr_state, loss, X, Y, Z = sess.run([model.logits, model.locs, model.scales, state, model.loss, model.X, model.Y, model.eval])
                plt.contour(X,Y,Z[:,:,0])
                plt.scatter(curr_state[0,:,0], curr_state[0,:,1])
                plt.savefig('data/test-{}.png'.format(i))
                plt.clf()
                print(scales[0])
                print(logits[0])
                print('i = {}, loss = {}'.format(i, loss))


    except KeyboardInterrupt:
        import ipdb; ipdb.set_trace()
        






def plot():
    ax = plt.gca(aspect='equal', xlim=W, ylim=H)
    rect = mpatches.Rectangle((0,0), W, H, color='C0')
    ax.add_patch(rect)

    dg = data_generator(N, key='shapes')
    objs = dg.__next__()

    for o in objs:
        o.plot(ax)
    plt.show()


if __name__ == "__main__":
    main()

    #while True:
    #    plot()