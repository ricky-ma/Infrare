import tensorflow as tf
from vae import VariationalAutoencoder
from pipeline import load_dataset


if __name__ == "__main__":
    # load training and validation data
    ds_train, ds_val = load_dataset()

    # load VAE model
    autoencoder = VariationalAutoencoder(input_dim=(None, None, 3)).autoencoder

    # calculate loss using initial distribution
    encoder = autoencoder.encoder_model()
    distribution_mean = tf.keras.layers.Dense(2, name='mean')(encoder)
    distribution_variance = tf.keras.layers.Dense(2, name='log_variance')(encoder)
    loss = autoencoder.get_loss(distribution_mean, distribution_variance)

    # compile and summarize model
    autoencoder.compile(loss=loss, optimizer='adam')
    autoencoder.summary()

    # fit
    autoencoder.fit(ds_train, ds_train, epochs=20, batch_size=64, validation_data=(ds_val, ds_val))


