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

flags.DEFINE_string('load_log_flags', None, help='')
flags.DEFINE_float('meta_step_final', 0.1, help='')
flags.DEFINE_integer('seed', 0, help='')
flags.DEFINE_string('env', 'HalfCheetahDir-v1', help='')
flags.DEFINE_bool('play', False, help='')
flags.DEFINE_bool('finetune', False, help='')
flags.DEFINE_integer('num_envs', 32, help='')
flags.DEFINE_float('lr', 5e-4, help='')
flags.DEFINE_float('eps', 0.2, help='')
flags.DEFINE_float('value_eps', 0.2, help='')
flags.DEFINE_integer('bs', 4096, help='')
flags.DEFINE_integer('horizon', 512, help='')
flags.DEFINE_integer('num_epochs', 15, help='')
flags.DEFINE_integer('num_fc', 2, help='')
flags.DEFINE_integer('fc_size', 100, help='')
flags.DEFINE_string('save_path', 'weights/model.weights', help='')
flags.DEFINE_string('load_path', 'weights/model.weights', help='')
flags.DEFINE_string('log_path', 'logs/{}'.format(datetime.datetime.now().strftime("openai-%Y-%m-%d-%H-%M-%S-%f")), help='')
flags.DEFINE_string('log_format', 'stdout,log,csv,tensorboard', help='')

# NOTE: must call one of the FLAGS for the dictionarify to work
os.environ['OPENAI_LOGDIR'] = FLAGS.log_path
os.environ['OPENAI_LOG_FORMAT'] = FLAGS.log_format

# dotdict-ing this was causing an error
FLAGS = {key: val.value for key, val in FLAGS._flags().items()}
if FLAGS['value_eps'] == 0.0: FLAGS['value_eps'] = None
