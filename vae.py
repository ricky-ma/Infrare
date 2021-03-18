import tensorflow as tf


class VariationalAutoencoder(object):
    # Model initialization
    def __init__(self, input_dim):
        self.input_dim = input_dim

        input_data = tf.keras.layers.Input(shape=input_dim)
        self.encoder_model = tf.keras.Model(input_data, self.encoder_layers(input_data))
        print(self.encoder_model.summary())

        self.decoder_model = tf.keras.Model(self.decoder_layers())
        print(self.decoder_model.summary())

        self.autoencoder = tf.keras.models.Model(input_data, self.decoder_model)
        print(self.autoencoder.summary())

    @staticmethod
    def sample_latent_features(distribution):
        distribution_mean, distribution_variance = distribution
        batch_size = tf.shape(distribution_variance)[0]
        random = tf.keras.backend.random_normal(shape=(batch_size, tf.shape(distribution_variance)[1]))
        return distribution_mean + tf.exp(0.5 * distribution_variance) * random

    def encoder_layers(self, input_data):
        encoder = tf.keras.layers.Conv2D(64, (5, 5), activation='relu')(input_data)
        encoder = tf.keras.layers.MaxPooling2D((2, 2))(encoder)
        encoder = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(encoder)
        encoder = tf.keras.layers.MaxPooling2D((2, 2))(encoder)
        encoder = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(encoder)
        encoder = tf.keras.layers.MaxPooling2D((2, 2))(encoder)
        encoder = tf.keras.layers.Flatten()(encoder)
        encoder = tf.keras.layers.Dense(16)(encoder)

        distribution_mean = tf.keras.layers.Dense(2, name='mean')(encoder)
        distribution_variance = tf.keras.layers.Dense(2, name='log_variance')(encoder)
        latent_encoding = tf.keras.layers.Lambda(self.sample_latent_features)(
            [distribution_mean, distribution_variance])
        return latent_encoding

    def decoder_layers(self):
        decoder_input = tf.keras.layers.Input(shape=2)
        decoder = tf.keras.layers.Dense(64)(decoder_input)
        decoder = tf.keras.layers.Reshape((1, 1, 64))(decoder)
        decoder = tf.keras.layers.Conv2DTranspose(64, (3, 3), activation='relu')(decoder)

        decoder = tf.keras.layers.Conv2DTranspose(64, (3, 3), activation='relu')(decoder)
        decoder = tf.keras.layers.UpSampling2D((2, 2))(decoder)

        decoder = tf.keras.layers.Conv2DTranspose(64, (3, 3), activation='relu')(decoder)
        decoder = tf.keras.layers.UpSampling2D((2, 2))(decoder)

        decoder_output = tf.keras.layers.Conv2DTranspose(1, (5, 5), activation='relu')(decoder)
        return decoder_output

    def get_loss(self, distribution_mean, distribution_variance):
        def get_reconstruction_loss(y_true, y_pred):
            reconstruction_loss = tf.keras.losses.mse(y_true, y_pred)
            reconstruction_loss_batch = tf.reduce_mean(reconstruction_loss)
            return reconstruction_loss_batch * 28 * 28

        def get_kl_loss(dmean, dvar):
            kl_loss = 1 + dvar - tf.square(dmean) - tf.exp(dvar)
            kl_loss_batch = tf.reduce_mean(kl_loss)
            return kl_loss_batch * (-0.5)

        def total_loss(y_true, y_pred):
            reconstruction_loss_batch = get_reconstruction_loss(y_true, y_pred)
            kl_loss_batch = get_kl_loss(distribution_mean, distribution_variance)
            return reconstruction_loss_batch + kl_loss_batch

        return total_loss
