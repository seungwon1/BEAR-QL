import tensorflow as tf
import numpy as np
import gzip, pickle
from matplotlib import pyplot as plt
import os, random

def get_session():
    tf.reset_default_graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    return session

def set_seed(seed_number, env):
    os.environ['PYTHONHASHSEED']=str(seed_number)
    random.seed(seed_number)
    np.random.seed(seed_number)
    tf.compat.v1.set_random_seed(seed_number)
    env.seed(seed_number)
    
def gs_kerl(x, y, dim_args, sigma = 20.0):
    # x, y: 2-D tensor, each of shape (batch_size * m, action_dims)
    batch_size, ac_dim = dim_args
    xx = tf.reshape(x, [batch_size, -1, ac_dim])
    yy = tf.reshape(y, [batch_size, -1, ac_dim])
    kxy = tf.square(tf.expand_dims(xx, axis = 2) - tf.expand_dims(yy, axis = 1))
    kxy = tf.reduce_sum(tf.exp(-tf.reduce_sum(kxy, axis = -1)/(2*(sigma)**2)), axis = (1,2))
    return kxy

def lp_kerl(x, y, dim_args, sigma = 10.0):
    # x, y: 2-D tensor, each of shape (batch_size * m, action_dims)
    batch_size, ac_dim = dim_args
    xx = tf.reshape(x, [batch_size, -1, ac_dim])
    yy = tf.reshape(y, [batch_size, -1, ac_dim])
    kxy = tf.abs(tf.expand_dims(xx, axis = 2) - tf.expand_dims(yy, axis = 1))
    kxy = tf.reduce_sum(tf.exp(-tf.reduce_sum(kxy, axis = -1)/(sigma)), axis = (1,2))
    return kxy

def mm_distance(x,y, num_sample, dim_args, flags = None):
    # x, y: 2-D tensor, each of shape (batch_size * m, action_dims)
    sigma = flags.sigma
    #if flags.game == 'Walker2d' or 'Ant':
    #    sigma = 20.0
    
    n, m = num_sample
    
    if flags.kernel == 'lp':
        mmd = lp_kerl(x, x, dim_args, sigma)/(n**2) - 2*lp_kerl(x, y, dim_args, sigma)/(n*m) + lp_kerl(y, y, dim_args, sigma)/(m**2)
    else:
        mmd = gs_kerl(x, x, dim_args, sigma)/(n**2) - 2*gs_kerl(x, y, dim_args, sigma)/(n*m) + gs_kerl(y, y, dim_args, sigma)/(m**2)
    return mmd
    
class replay_buffer(object):
    
    def __init__(self, batch_size=256):
        self.batch_size = batch_size
        
    def load_data(self, data_path):
        with gzip.open(data_path, 'rb') as f:
            replay_buffer = pickle.load(f)
            
        self.state = np.concatenate(replay_buffer[:,0]).ravel().reshape(replay_buffer.shape[0],-1).astype('float32')
        self.next_state = np.concatenate(replay_buffer[:,1]).ravel().reshape(replay_buffer.shape[0],-1).astype('float32')
        self.action = np.concatenate(replay_buffer[:,2]).ravel().reshape(replay_buffer.shape[0],-1).astype('float32')
        self.reward = replay_buffer[:,3].astype('float32')
        self.done = replay_buffer[:,4].astype('float32')
        replay_buffer = 0
        
    def sample(self, batch_size):
        idx = np.random.randint(self.state.shape[0],size=batch_size)
        s, ns, a, r, d = self.state[idx], self.next_state[idx], self.action[idx], self.reward[idx], self.done[idx]
        return s, ns, a, r, d