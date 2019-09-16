import tensorflow as tf
from six.moves import cPickle
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil


class RNN:
    '''
    <Configuration example>
    config = {
        'ID' : 'test_NN',
        'n_iter' : 5000,
        'n_prt' : 100,
        'n_input' : 28,
        'n_step' : 14,
        'n_output' : 28,
        'n_batch' : 50,
        'n_save' : 1000,
        'n_history' : 50,
        'LR' : 0.0002
    }
    '''
    def __init__(self, config):
        self.ID = config['ID']
        self.n_iter = config['n_iter']
        self.n_prt = config['n_prt']
        self.n_input = config['n_input']
        self.n_step = config['n_step']
        self.n_output = config['n_output']
        self.n_batch = config['n_batch']
        self.n_save = config['n_save']
        self.n_history = config['n_history']
        self.LR = config['LR']
        self.history = {
            'train' : [],
        }
        
        self.checkpoint = 0
        self.path = './{}'.format(self.ID)
        try: 
            os.mkdir(self.path)
            os.mkdir('{0}/{1}'.format(self.path, 'checkpoint'))
        except FileExistsError:
            msg = input('[FileExistsError] Will you remove directory? [Y/N] ')
            if msg == 'Y': # or debug 
                shutil.rmtree(self.path)
                os.mkdir(self.path)
                os.mkdir('{0}/{1}'.format(self.path, 'checkpoint'))
            else: 
                print('Please choose another ID')
                assert 0
                  
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.x = tf.placeholder(tf.float32, [None, self.n_step, self.n_input], name='x')
            self.y = tf.placeholder(tf.float32, [None, self.n_output], name='y')
            
            self.feature = self.feature_map(self.x)
            self.pred = self.clf(self.feature['hidden1'])

            self.loss = self.compute_loss(self.pred, self.y)
            self.optm = tf.train.AdamOptimizer(learning_rate=self.LR).minimize(self.loss)
            self.init = tf.global_variables_initializer()
            self.saver = tf.train.Saver(max_to_keep=None)
        
        self.sess = tf.Session(graph=self.graph, config=tf.ConfigProto(allow_soft_placement=True))
        self.sess.run(self.init)
        
        print('Model ID : {}'.format(self.ID))
        print('Model saved at : {}'.format(self.path))

    ## Layers
    def fully_connected_layer(self, input_tensor, name, n_out, activation_fn=tf.nn.relu):
        n_in = input_tensor.get_shape()[-1].value
        with tf.variable_scope(name):
            weight = tf.get_variable('weight', [n_in, n_out], tf.float32)
            bias = tf.get_variable('bias', [n_out], tf.float32, tf.constant_initializer(0.0))
            logits = tf.add(tf.matmul(input_tensor, weight), bias, name='logits')
        if activation_fn is None : return logits
        else : return activation_fn(logits, name='activation')

    def lstm_layer(self, input_tensor, name, n_out):
        with tf.variable_scope('rnn'):
            with tf.variable_scope(name):
                lstm = tf.contrib.rnn.BasicLSTMCell(n_out, forget_bias=1.0)
                h, c = tf.nn.dynamic_rnn(lstm, input_tensor, dtype=tf.float32)
        return h, c
        
    ## Feature map
    def feature_map(self, x):
        with tf.variable_scope('feature_map'):
            lstm1_h, lstm1_c = self.lstm_layer(x, 'lstm1_h', 128)
            lstm2_h, lstm2_c = self.lstm_layer(lstm1_h, 'lstm2_h', 256)
            hidden1 = self.fully_connected_layer(lstm2_h[:,-1,:], 'hidden1', 100)
        return {
            'lstm1_h' : lstm1_h,
            'lstm1_c' : lstm1_c,
            'lstm2_h' : lstm2_h,
            'lstm2_c' : lstm2_c,
            'hidden1' : hidden1
        }
    
    ## Compute loss
    def compute_loss(self, pred, y):
        with tf.variable_scope('compute_loss'):
            loss = tf.square(tf.subtract(pred, y))
            loss = tf.reduce_mean(loss)
        return loss

    ## Classifier
    def clf(self, feature):
        with tf.variable_scope('clf'):
            pred = self.fully_connected_layer(feature, 'pred', self.n_output, None)
        return pred
    
    ## Train
    def fit(self, data):
        for epoch in range(1, self.n_iter+1):
            train_x, train_y = data.train.next_batch(self.n_batch)
            train_x = train_x.reshape(-1, 28, 28)
            
            for j in range(self.n_step):
                self.sess.run(self.optm, feed_dict={self.x: train_x[:,j:j+self.n_step,:],  self.y: train_x[:,j+self.n_step]})
            
            if epoch % self.n_prt == 0:
                train_loss = self.get_loss(train_x[:,13:13+self.n_step,:], train_x[:,13+self.n_step])
                print('Your loss ({0}/{1}) : {2}'.format(epoch, self.n_iter, train_loss))
                
            if epoch % self.n_save == 0:
                self.checkpoint += self.n_save
                self.save('{0}/{1}/{2}_{3}'.format(self.path, 'checkpoint', self.ID, self.checkpoint))
            
            if epoch % self.n_history == 0:
                test_x, test_y = data.test.next_batch(self.n_batch)
                train_loss = self.get_loss(train_x[:,13:13+self.n_step,:], train_x[:,13+self.n_step])
                self.history['train'].append(train_loss)
    
    ## Predict
    def predict(self, x):
        gen_img = []
        sample = x.copy()
        feeding_img = x.copy()
        
        for i in range(self.n_step):
            test_pred = self.sess.run(self.pred, feed_dict={self.x: feeding_img.reshape(1, 14, 28)})
            feeding_img = np.delete(feeding_img, 0, 0)
            feeding_img = np.vstack([feeding_img, test_pred])
            gen_img.append(test_pred)
        
        for i in range(self.n_step):
            sample = np.vstack([sample, gen_img[i]])
        
        return sample
    
    ## Analysis
    def get_feature(self, x):
        feature = self.sess.run(self.feature, feed_dict={self.x : x})
        return feature
    
    def get_loss(self, x, y):
        loss = self.sess.run(self.loss, feed_dict={self.x : x, self.y : y})
        return loss
    
    ## Save/Restore
    def save(self, path):
        self.saver.save(self.sess, path)
        
    def load(self, path):
        self.saver.restore(self.sess, path)
        checkpoint = path.split('_')[-1]
        self.checkpoint = int(checkpoint)
        print('Model loaded from file : {}'.format(path))
        