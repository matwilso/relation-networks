import numpy as np
import tensorflow as tf
from rns.constant import W, H, R, DIRS
from rns.shapes import SAMPLERS, make_image

def subsample_postbatch(state, FLAGS):
    backset = tf.cast((FLAGS['subsample']+1)*tf.random.uniform([]), tf.int32)
    state['state'] = state['state'][:,:FLAGS['num_shapes']-backset]
    return state
    #return tf.gather(state, tf.range(idx), axis=1)

def to_float(state):
    state['state'] = tf.to_float(state['state'])
    return state

def normalize(state):
    state['state'] = (state['state'] - W//2) / (W//2)
    #state['image'] = (state['image'] / 0.5) - 1.0
    return state

#def preproc_image(img, width=224, height=224, dtype=np.uint8):
#    """Crop and resize image to the given square shape"""
#    # reference: https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py 
#    h,w,c = img.shape
#    crop = (w - h) // 2
#    if crop != 0:
#        img = img[:, crop:-crop]
#    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
#    return img.astype(dtype)

def resize_image(state):
    state['image'] = tf.image.resize_images(state['image'], size=[64,64])
    return state

def data_generator(n, samplers):
    samplers = [SAMPLERS[key] for key in samplers]
    while True:
        sampler = np.random.choice(samplers)
        raw_data = sampler(n)
        samples = {}
        samples['state'] = raw_data['state']
        samples['image'] = make_image(raw_data['shapes'])
        yield samples

class Dataset(object):
    def __init__(self, FLAGS):
        dg = lambda : data_generator(FLAGS['num_shapes'], FLAGS['samplers'])

        self.dataset = tf.data.Dataset.from_generator(dg, {'state': tf.int64, 'image': tf.float32}, {'state': tf.TensorShape([None,2]), 'image':tf.TensorShape([None,None,1])})
        self.dataset = self.dataset.map(to_float)
        self.dataset = self.dataset.map(resize_image)
        self.dataset = self.dataset.batch(FLAGS['bs'])
        self.dataset = self.dataset.map(normalize)
        if FLAGS['subsample'] > 0:
            self.dataset = self.dataset.map(lambda x: subsample_postbatch(x, FLAGS))
        self.dataset = self.dataset.prefetch(10)

        self.iterator = self.dataset.make_one_shot_iterator()
        self.state = self.iterator.get_next()
