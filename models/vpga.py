import tensorflow as tf
from tensorflow_probability import distributions as tfd
import numpy as np


class VPGA(tf.keras.Model):
    """Convolutional perceptual generative autoencoder."""

    def __init__(self, latent_dim, input_shape, zn_rec_coeff, zh_rec_coeff, vrec_coeff, vkld_coeff):
        super(VPGA, self).__init__()
        self.architecture = "VPGA"
        self.batch_size = input_shape[0]
        self.latent_dim = latent_dim
        self.zn_rec_coeff = zn_rec_coeff
        self.zh_rec_coeff = zh_rec_coeff
        self.vrec_coeff = vrec_coeff
        self.vkld_coeff = vkld_coeff
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=input_shape),
                # tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.float32)),
                tf.keras.layers.Conv2D(
                    filters=32, kernel_size=3, strides=2, activation='relu'),
                tf.keras.layers.Conv2D(
                    filters=64, kernel_size=3, strides=2, activation='relu'),
                tf.keras.layers.Conv2D(
                    filters=128, kernel_size=3, strides=2, activation='relu'),
                tf.keras.layers.Conv2D(
                    filters=256, kernel_size=4, strides=2, activation='relu'),
                tf.keras.layers.Flatten(),
                # No activation
                tf.keras.layers.Dense(latent_dim + latent_dim),
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

    def enc_dec(self, x):
        # encode
        z_mu, z_log_sigma_sq = self.encode(x)
        # decode
        img_rec = self.decode(z_mu, True)
        return z_mu, z_log_sigma_sq, img_rec

    def dec_enc(self, z, no_enc_grad=False):
        # decode
        img = self.decode(z, True)
        # encode
        z_rec, _ = self.encode(img)
        if no_enc_grad:
            z_rec -= self.encode(tf.stop_gradient(img))[0] - tf.stop_gradient(z_rec)
        return z_rec

    def compute_loss(self, x):
        normal_dist = tfd.MultivariateNormalDiag(scale_diag=np.ones([self.latent_dim]))
        z_mu, z_log_sigma_sq, img_rec = self.enc_dec(x)
        z_noise = tf.exp(0.5 * z_log_sigma_sq) * tf.random.normal(tf.shape(z_mu))
        zn_targ, zh_targ = normal_dist.sample(self.batch_size), tf.stop_gradient(z_mu)
        zn_rec, zh_rec = self.dec_enc(zn_targ), self.dec_enc(zh_targ)
        z_mu_rec, z_rec = self.dec_enc(tf.stop_gradient(z_mu), True), \
                          self.dec_enc(tf.stop_gradient(z_mu) + z_noise, True)

        # Image reconstruction loss
        img_rec_loss = tf.keras.losses.MSE(x, img_rec)
        # Latent reconstruction loss under a multivariate normal distribution, N
        zn_rec_loss = tf.keras.losses.MSE(zn_targ, zn_rec)
        # Latent reconstruction loss under a latent-target distribution, H
        zh_rec_loss = tf.keras.losses.MSE(zh_targ, zh_rec)
        z_mu_norm = tf.stop_gradient(tf.sqrt(tf.reduce_mean(tf.square(z_mu), 0)))
        # VAE reconstruction loss
        vrec_loss = tf.keras.losses.MSE(z_mu_rec / z_mu_norm, z_rec / z_mu_norm)
        # VAE KL divergence loss
        vkld_loss = -tf.reduce_mean(0.5 * (1 + z_log_sigma_sq - z_mu ** 2 - tf.exp(z_log_sigma_sq)))

        # TODO: fix loss calculation
        latent_loss = (self.zn_rec_coeff * zn_rec_loss) + (self.zh_rec_coeff * zh_rec_loss)
        vae_loss = (self.vrec_coeff * vrec_loss) + (self.vkld_coeff * vkld_loss)

        enc_loss = img_rec_loss + latent_loss + vae_loss
        dec_loss = img_rec_loss + self.vrec_coeff * vrec_loss
        return enc_loss, dec_loss
