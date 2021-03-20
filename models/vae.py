import tensorflow as tf
import math


class VAE(tf.keras.Model):
    """Convolutional variational autoencoder."""

    def __init__(self, latent_dim, intermediate_dim, input_shape):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=input_shape),
                tf.keras.layers.BatchNormalization(),

                # fully connected dense layer 1
                tf.keras.layers.Dense(intermediate_dim),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation('relu'),
                tf.keras.layers.Dropout(0.2),

                # fully connected dense layer 2
                tf.keras.layers.Dense(intermediate_dim),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation('relu'),
                tf.keras.layers.Dropout(0.2),

                # fully connected dense layer 3
                tf.keras.layers.Dense(intermediate_dim),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation('relu'),
                tf.keras.layers.Dropout(0.2),

                # z_mean 1
                tf.keras.layers.Dense(intermediate_dim),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation('relu'),
                tf.keras.layers.Dropout(0.2),

                # z_mean 2
                tf.keras.layers.Dense(intermediate_dim),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation('relu'),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(latent_dim),

                # z_log_var_1
                tf.keras.layers.Dense(intermediate_dim),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation('relu'),
                tf.keras.layers.Dropout(0.2),

                # z_log_var_2
                tf.keras.layers.Dense(intermediate_dim),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation('relu'),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(latent_dim),
            ]
        )
        print(self.encoder.summary())
        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                # fully connected dense layer 1
                tf.keras.layers.Dense(intermediate_dim),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation('relu'),
                tf.keras.layers.Dropout(0.2),
                # fully connected dense layer 2
                tf.keras.layers.Dense(intermediate_dim),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation('relu'),
                tf.keras.layers.Dropout(0.2),
                # fully connected dense layer 3
                tf.keras.layers.Dense(intermediate_dim),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation('relu'),
                tf.keras.layers.Dropout(0.2),
                # fully connected dense layer 4
                tf.keras.layers.Dense(intermediate_dim),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation('relu'),
                tf.keras.layers.Dropout(0.2),
                # fully connected dense layer 5
                tf.keras.layers.Dense(intermediate_dim),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation('relu'),
                tf.keras.layers.Dropout(0.2),
                # decoded mean
                tf.keras.layers.Dense(units=64 * 64 * 32),
                tf.keras.layers.Dense(units=64 * 64 * 32),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation('relu'),
                tf.keras.layers.Dropout(0.2),
                # decoded variance
                tf.keras.layers.Dense(units=64 * 64 * 32, activation=tf.nn.relu),
                tf.keras.layers.Reshape(target_shape=(64, 64, 32)),
            ]
        )
        print(self.decoder.summary())

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits
