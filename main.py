import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import itertools
import tensorflow as tf


BS = 30
N = 10
W = 64  
H = 64  
R = 2
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
    border = W // 8
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
    border = W // 8

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

def data_generator(n):
    while True:
        sampler = np.random.choice([uniform, cluster1, cluster2])
        samples = sampler(n)
        yield samples['state']


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

class Model(object):
    def __init__(self, state):
        def do_g_sum(state):
            # TODO: try making this part parallel too and see if it makes it faster
            x = tf.range(tf.shape(state)[0], dtype=tf.int32)
            a, b = x[None, :, None], x[:, None, None]
            cartesian_product = tf.concat([b + tf.zeros_like(a), tf.zeros_like(b) + a], axis = 2)
            #new_shape = tf.stack([-1, tf.shape(cartesian_product)[-1]])
            #cartesian_product = tf.reshape(cartesian_product, new_shape)
            cartesian_product = tf.reshape(cartesian_product, [-1])

            ijs = tf.reshape(tf.gather(state, cartesian_product), [-1,2,2])
            ijs = tf.concat([ijs[:,0], ijs[:,1]], axis=1)
            g = relation(ijs)
            return tf.reduce_sum(g, axis=0)
        g_sum = tf.map_fn(do_g_sum, state, dtype=tf.float32)




def main():
    dg = lambda : data_generator(N)
    ds = tf.data.Dataset.from_generator(dg, tf.int64, tf.TensorShape([None,2]))
    ds = ds.batch(BS)
    ds = ds.map(to_float)
    ds = ds.map(normalize)
    ds = ds.prefetch(10)

    iterator = ds.make_one_shot_iterator()
    state = iterator.get_next()

    m = Model(state)


    sess = tf.InteractiveSession()























def plot():
    ax = plt.gca(aspect='equal', xlim=W, ylim=H)
    rect = mpatches.Rectangle((0,0), W, H, color='C0')
    ax.add_patch(rect)

    dg = data_generator(N)
    objs = dg.__next__()

    for o in objs['og']:
        o.plot(ax)
    plt.show()


if __name__ == "__main__":
    main()

    #while True:
    #    plot()