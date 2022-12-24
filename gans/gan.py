from __future__ import print_function, division
import numpy as np
import cv2

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras import Input
from tensorflow.keras.layers import InputLayer, Dense
from tensorflow.keras.models import Sequential, Model
from tensorflow_addons.layers import Maxout
from tensorflow.keras.optimizers import Adam

"""
https://github.com/goodfeli/adversarial/blob/master/mnist.yaml
https://pylearn2.readthedocs.io/en/latest/theano_to_pylearn2_tutorial.html
https://pylearn2.readthedocs.io/en/latest/yaml_tutorial/index.html#yaml-tutorial
"""
class Generator(Model):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()

        self.model = Sequential(
            layers = [
                InputLayer(input_shape=input_dim),
                Dense(1200, activation='relu'),
                Dense(1200, activation='relu'),
                Dense(output_dim, activation='tanh'),
            ],
            name = 'Generator'
        )

        self.model.summary()
    
    def build(self, input_shape):
        super(Generator, self).build(input_shape)

    def call(self, inputs, training=False):
        return self.model(inputs, training=training)

class Discriminator(Model):
    def __init__(self, input_dim, output_dim):
        super(Discriminator, self).__init__()

        self.model = Sequential(
            layers = [
                InputLayer(input_shape=input_dim),
                Dense(1200),
                Maxout(240),
                Dense(1200),
                Maxout(240),            
                Dense(output_dim, activation='sigmoid'),
            ],
            name = 'Discriminator'
        )

        self.model.summary()
    
    def build(self, input_shape):
        super(Discriminator, self).build(input_shape)

    def call(self, inputs, training=False):
        return self.model(inputs, training=training)

class GAN():
    def __init__(self, steps=1000, batch_size=32, sample_intervel=50):
        self.steps = steps
        self.sample_intervel = sample_intervel
        self.batch_size = batch_size
        self.g_input_dim = 100
        self.g_output_dim = 28*28
        self.d_input_dim = self.g_output_dim
        self.d_output_dim = 1

        self.optimizer = Adam(2e-4, 0.5)

        self.discriminator = Discriminator(self.d_input_dim, self.d_output_dim)
        self.discriminator.compile(optimizer=self.optimizer, loss='binary_crossentropy')

        self.generator = Generator(self.g_input_dim, self.g_output_dim)
        self.generator_discriminator = Sequential()
        self.generator_discriminator.add(self.generator)
        self.discriminator.trainable = False
        self.generator_discriminator.add(self.discriminator)
        self.generator_discriminator.compile(optimizer=self.optimizer, loss='binary_crossentropy')

        self.writer = tf.summary.create_file_writer('./runs/')
    
    def train(self):
        # load training dataset
        train_data = mnist.load_data()[0][0]
        b, h, w = train_data.shape
        train_data = train_data.reshape(b, h*w)
        train_data = train_data / 127.5 - 1.0

        real = np.ones((self.batch_size, 1))
        fake = np.zeros((self.batch_size, 1))

        for step in range(self.steps):
            # make training data for discriminator
            idx = np.random.randint(0, train_data.shape[0], self.batch_size)
            images = train_data[idx]
            noise = np.random.normal(0, 1, (self.batch_size, self.g_input_dim))
            g_images = self.generator(noise, training=False)
            # train discriminator
            d_loss_real = self.discriminator.train_on_batch(images, real)
            d_loss_fake = self.discriminator.train_on_batch(g_images, fake)
            d_loss = 0.5 * (d_loss_fake + d_loss_real)

            # make training data for generator
            noise = np.random.normal(0, 1, (self.batch_size, self.g_input_dim))
            # train generator
            g_loss = self.generator_discriminator.train_on_batch(noise, real)

            with self.writer.as_default():
                tf.summary.scalar('loss/g_loss', g_loss, step)
                tf.summary.scalar('loss/d_loss', d_loss, step)

            if step % self.sample_intervel == 0:
                sample_row = 4
                sample_col = 4
                sample_count = sample_row * sample_col
                samples = np.zeros((sample_row * 28, sample_col * 28))
                noise = np.random.normal(0, 1, (sample_count, self.g_input_dim))
                g_images = self.generator(noise, training=False).numpy()
                g_images = 0.5 * (g_images + 1.0)
                g_images = np.clip(g_images*255, 0, 255)

                for r in range(sample_row):
                    for c in range(sample_col):
                        samples[r*28:(r+1)*28, c*28:(c+1)*28] = g_images[r*sample_col+c].reshape(28, 28)
                sample_name = 'images/{}.png'.format(step)
                cv2.imwrite(sample_name, samples)

    def test(self):
        pass

if __name__ == '__main__':
    print('\nStart to train GAN...\n')
    gan = GAN(steps=2000, batch_size=64, sample_intervel=100)
    gan.train()
    print('\nTraining GAN finished!\n')