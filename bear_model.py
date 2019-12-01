import tensorflow as tf
import numpy as np

class BEAR_model(object):
    def __init__(self, num_actions, max_action):
        self.num_actions = num_actions
        self.max_action = max_action
        
    def critic(self, obs, act, network, k = 2, reuse=False):
        with tf.variable_scope(network, reuse=reuse):
            with tf.variable_scope('fc1'):
                fc = tf.contrib.layers.fully_connected(tf.concat([obs, act], axis=1), 400, activation_fn=tf.nn.relu)
                fc = tf.contrib.layers.fully_connected(fc, 300, activation_fn=tf.nn.relu)
                fc = tf.contrib.layers.fully_connected(fc, 1, activation_fn=None)
                
            with tf.variable_scope('fc2'):
                fc2 = tf.contrib.layers.fully_connected(tf.concat([obs, act], axis=1), 400, activation_fn=tf.nn.relu)
                fc2 = tf.contrib.layers.fully_connected(fc2, 300, activation_fn=tf.nn.relu)
                fc2 = tf.contrib.layers.fully_connected(fc2, 1, activation_fn=None)
                
        return tf.concat([fc, fc2], axis = 1)
    
    def actor(self, obs, network, size = 1, reuse=False):
        if size != 1:
            obs = tf.tile(obs, [size, 1])
            
        with tf.variable_scope(network, reuse=reuse):
            with tf.variable_scope('fc'):
                fc = tf.contrib.layers.fully_connected(obs, 400, activation_fn=tf.nn.relu)
                fc = tf.contrib.layers.fully_connected(fc, 300, activation_fn=tf.nn.relu)
                act_mean = tf.contrib.layers.fully_connected(fc, self.num_actions, activation_fn=None)
                act_logstd = tf.contrib.layers.fully_connected(fc, self.num_actions, activation_fn=None)
                act_std = tf.exp(act_logstd)
                
            action = act_mean + act_std * tf.clip_by_value(tf.random.normal(shape = tf.shape(act_std), mean= 0.0, stddev=1.0, dtype=tf.dtypes.float32), -0.5, 0.5)        
        return self.max_action * tf.nn.tanh(action), action

class VAE(object):
    def __init__(self, latent_dim, max_action, act_dim, batch_size):
        self.latent_dim = latent_dim
        self.max_action = max_action
        self.act_dim = act_dim
        self.batch_size = batch_size
        
    def encoder(self, obs, act, network = 'encoder', reuse = False):
        with tf.variable_scope(network, reuse=reuse):
            with tf.variable_scope('fc'):
                fc = tf.contrib.layers.fully_connected(tf.concat([obs, act], axis=1), 750, activation_fn=tf.nn.relu)
                fc = tf.contrib.layers.fully_connected(fc, 750, activation_fn=tf.nn.relu)
                act_mean = tf.contrib.layers.fully_connected(fc, self.latent_dim, activation_fn=None)
                act_logstd = tf.clip_by_value(tf.contrib.layers.fully_connected(fc, self.latent_dim, activation_fn=None), -4, 15)
                std = tf.exp(act_logstd)
        
        z = act_mean + tf.random.normal(shape = tf.shape(std), mean= 0.0, stddev=std, dtype=tf.dtypes.float32)
        recon_vec, _ = self.decoder(obs = obs, z = z)
        return (recon_vec, act_mean, std)
        
    def decoder(self, obs, size = 1, z = None, network = 'decoder', reuse = False):
        if size != 1:
            obs = tf.tile(obs, [size, 1])
        
        if z is None:
            mean = tf.constant(0.0, shape= [self.batch_size*size, self.latent_dim])
            std =  tf.constant(1.0, shape= [self.batch_size*size, self.latent_dim])
            z = tf.clip_by_value(mean + tf.random.normal(shape = tf.shape(std), mean= 0.0, stddev=std, dtype=tf.dtypes.float32), -0.5, 0.5)
        
        with tf.variable_scope(network, reuse=reuse):
            with tf.variable_scope('fc'):
                fc = tf.contrib.layers.fully_connected(tf.concat([obs, z], axis=1), 750, activation_fn=tf.nn.relu)
                fc = tf.contrib.layers.fully_connected(fc, 750, activation_fn=tf.nn.relu)
                fc = tf.contrib.layers.fully_connected(fc, self.act_dim, activation_fn=None)
        return self.max_action * tf.nn.tanh(fc), fc
    
