import numpy as np
import tensorflow as tf
from rns.constant import W, H, R, DIRS
from rns.shapes import uniform, cluster1, cluster2, make_image
from rns.viz import plot_shapes, plot_arr

def subsample_postbatch(state, FLAGS):
    backset = tf.cast((FLAGS['subsample']+1)*tf.random.uniform([]), tf.int32)
    return state[:,:FLAGS['num_shapes']-backset]
    #return tf.gather(state, tf.range(idx), axis=1)

def to_float(state):
    return tf.to_float(state)

def to_state(state):
    return state['state']

def normalize(state):
    return (state - W//2) / (W//2)

def data_generator(n):
    while True:
        sampler = np.random.choice([uniform, cluster1, cluster2])
        samples = {}
        raw_data = sampler(n)
        samples['state'] = raw_data['state']
        samples['image'] = make_image(raw_data['shapes'])
        yield samples

class Dataset(object):
    def __init__(self, FLAGS):
        dg = lambda : data_generator(FLAGS['num_shapes'])

        self.dataset = tf.data.Dataset.from_generator(dg, {'state': tf.int64, 'image': tf.float32}, {'state': tf.TensorShape([None,2]), 'image':tf.TensorShape([None,None])})

        self.dataset = self.dataset.map(to_state)
        self.dataset = self.dataset.map(to_float)
        self.dataset = self.dataset.batch(FLAGS['bs'])
        self.dataset = self.dataset.map(normalize)
        self.dataset = self.dataset.map(lambda x: subsample_postbatch(x, FLAGS))
        self.dataset = self.dataset.prefetch(10)

        self.iterator = self.dataset.make_one_shot_iterator()
        self.state = self.iterator.get_next()
