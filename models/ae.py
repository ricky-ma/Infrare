import tensorflow as tf
import numpy as np


class AE(tf.keras.Model):
    """Autoencoder."""

    def __init__(self, latent_dim, input_shape):
        super(AE, self).__init__()
        self.architecture = "AE"
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=input_shape),
                tf.keras.layers.Reshape(target_shape=(np.prod(input_shape),)),
                tf.keras.layers.Dense(latent_dim, activation='relu'),
                # No activation
                tf.keras.layers.Dense(latent_dim),
            ]
        )
        print(self.encoder.summary())
        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(latent_dim,)),
                tf.keras.layers.Dense(latent_dim, activation='relu'),
                tf.keras.layers.Dense(units=256 * 256 * 3),
                tf.keras.layers.Reshape(target_shape=input_shape),
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
