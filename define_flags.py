"""
Command-line argument parsing.
"""
import os
import datetime
import tensorflow as tf 
from tensorflow.python.platform import flags
FLAGS = flags.FLAGS

#class dotdict(dict):
#    """dot.notation access to dictionary attributes"""
#    __getattr__ = dict.get
#    __setattr__ = dict.__setitem__
#    __delattr__ = dict.__delitem__
flags.DEFINE_bool('vae', False, help='')

flags.DEFINE_string('log_dir', 'logs/', help='')
flags.DEFINE_float('lr', 3e-4, help='')
flags.DEFINE_integer('bs', 32, help='')
flags.DEFINE_integer('num_shapes', 15, help='')
flags.DEFINE_integer('num_vae_samples', 16, help='')
flags.DEFINE_integer('subsample', 0, help='')
flags.DEFINE_integer('k', 25, help='num components')
flags.DEFINE_integer('vae_k', 100, help='num components')
flags.DEFINE_integer('z_size', 32, help='')
flags.DEFINE_integer('vae_z_size', 32, help='')
flags.DEFINE_string('save_path', 'weights/model.weights', help='')
flags.DEFINE_string('load_path', 'weights/model.weights', help='')
flags.DEFINE_bool('plot_shapes', False, help='')
flags.DEFINE_bool('plot_samples', False, help='')
#flags.DEFINE_string('log_path', 'logs/{}'.format(datetime.datetime.now().strftime("openai-%Y-%m-%d-%H-%M-%S-%f")), help='')
flags.DEFINE_string('log_format', 'stdout,log,tensorboard', help='')
flags.DEFINE_string('samplers', 'uniform,cluster1,cluster2', help='')
flags.DEFINE_string('suffix', '', help='')
# NOTE: must call one of the FLAGS for the dictionarify to work
FLAGS.k

def _make_hp_str(FLAGS):
    hp_str = ''
    hp_str += 'vae/' if FLAGS['vae'] else 'objs/'
    hp_str += 'lr{:0.3E}'.format(FLAGS['lr']) 
    hp_str += '-bs{}'.format(FLAGS['bs']) 
    hp_str += '-nshapes{}'.format(FLAGS['num_shapes']) 
    hp_str += '-subsample{}'.format(FLAGS['subsample']) 
    hp_str += '-k{}'.format(FLAGS['k']) 
    hp_str += '-vaek{}'.format(FLAGS['vae_k']) 
    hp_str += '-zsize{}'.format(FLAGS['z_size']) 
    hp_str += '-vaesamples{}'.format(FLAGS['num_vae_samples']) 
    hp_str += '-data{}'.format('-'.join(FLAGS['samplers']))
    hp_str += '-suffix{}'.format(FLAGS['suffix']) if FLAGS['suffix'] else ''
    # TODO: add unique identifier
    return hp_str

# dotdict-ing this was causing an error
FLAGS = {key: val.value for key, val in FLAGS._flags().items()}

FLAGS['samplers'] = FLAGS['samplers'].split(',')

FLAGS['hp_str'] = _make_hp_str(FLAGS)

FLAGS['log_path'] = os.path.join(FLAGS['log_dir'], FLAGS['hp_str'])
os.makedirs(FLAGS['log_path'], exist_ok=True)
FLAGS['plot_path'] = os.path.join(FLAGS['log_path'], 'data/')
os.makedirs(FLAGS['plot_path'], exist_ok=True)
#os.environ['OPENAI_LOGDIR'] = FLAGS['log_path']
#os.environ['OPENAI_LOG_FORMAT'] = FLAGS['log_format']


print(FLAGS['hp_str'])


