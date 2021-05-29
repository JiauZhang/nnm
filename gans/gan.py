from __future__ import print_function, division
import numpy as np
import cv2

from tensorflow.keras.datasets import mnist
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
                Dense(output_dim, activation='sigmoid'),
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

        self.generator_discriminator = Sequential()        
        self.generator = Generator(self.g_input_dim, self.g_output_dim)
        self.generator_discriminator.add(self.generator)
        self.discriminator.trainable = False
        self.generator_discriminator.add(self.discriminator)
        self.generator_discriminator.compile(optimizer=self.optimizer, loss='binary_crossentropy')        
    
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
            g_loss = self.generator_discriminator.train_on_batch(noise, fake)

            print('step={}-->d_loss={}, g_loss={}'.format(step, d_loss, g_loss))

            if step % self.sample_intervel == 0:
                sample_count = 8
                noise = np.random.normal(0, 1, (sample_count, self.g_input_dim))
                g_images = self.generator(noise, training=False).numpy()
                g_images = 0.5 * g_images + 0.5
                g_images = np.clip(g_images, 0, 1) * 255

                for idx in range(sample_count):
                    sample_name = '{}_{}.png'.format(step, idx)
                    cv2.imwrite(sample_name, g_images[idx].reshape(28, 28))

    def test(self):
        pass

gan = GAN(steps=800, batch_size=64, sample_intervel=200)
gan.train()

print('\nTraining GAN finished!\n')