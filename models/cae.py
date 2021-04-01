import tensorflow as tf
import numpy as np


class CAE(tf.keras.Model):
    """Convolutional autoencoder."""

    def __init__(self, latent_dim, input_shape):
        super(CAE, self).__init__()
        self.architecture = "CVAE"
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=input_shape),
                tf.keras.layers.Conv2D(
                    filters=32, kernel_size=3, strides=2, activation='relu'),
                tf.keras.layers.Conv2D(
                    filters=64, kernel_size=3, strides=2, activation='relu'),
                tf.keras.layers.Conv2D(
                    filters=128, kernel_size=3, strides=2, activation='relu'),
                tf.keras.layers.Conv2D(
                    filters=256, kernel_size=4, strides=2, activation='relu'),
                tf.keras.layers.Flatten(),
                # tf.keras.layers.GlobalMaxPool2D(),
                # No activation
                tf.keras.layers.Dense(latent_dim),
            ]
        )
        print(self.encoder.summary())
        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                tf.keras.layers.Dense(units=16 * 16 * 32, activation=tf.nn.relu),
                tf.keras.layers.Reshape(target_shape=(16, 16, 32)),
                tf.keras.layers.Conv2DTranspose(
                    filters=256, kernel_size=4, strides=2, padding='same',
                    activation='relu'),
                tf.keras.layers.Conv2DTranspose(
                    filters=128, kernel_size=3, strides=2, padding='same',
                    activation='relu'),
                tf.keras.layers.Conv2DTranspose(
                    filters=64, kernel_size=3, strides=2, padding='same',
                    activation='relu'),
                tf.keras.layers.Conv2DTranspose(
                    filters=32, kernel_size=3, strides=2, padding='same',
                    activation='relu'),
                # No activation
                tf.keras.layers.Conv2DTranspose(
                    filters=3, kernel_size=3, strides=1, padding='same'),
            ]
        )
        print(self.decoder.summary())

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits

    def compute_loss(self, x):
        z = self.encode(x)
        x_logit = self.decode(z)
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
        logpx_z = tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        return logpx_z
