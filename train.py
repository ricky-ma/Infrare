import tensorflow as tf
import time
import numpy as np
import matplotlib.pyplot as plt
from model import CVAE
from pipeline import load_dataset
from tqdm import tqdm


def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis)


def compute_loss(model, x):
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)
    return -tf.reduce_mean(logpx_z + logpz - logqz_x)


@tf.function
def train_step(model, x, optimizer):
    """
    Executes one training step and returns the loss.
    This function computes the loss and gradients, and uses the latter to
    update the model's parameters.
    """
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


def generate_and_save_images(model, epoch, test_sample):
    mean, logvar = model.encode(test_sample)
    z = model.reparameterize(mean, logvar)
    predictions = model.sample(z)
    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0], cmap='gray')
        plt.axis('off')

    # tight_layout minimizes the overlap between 2 sub-plots
    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()


if __name__ == "__main__":
    model_dir = "vae/"
    batch_size = 32
    epochs = 10
    latent_dim = 2  # set the dimensionality of the latent space to a plane for visualization later
    num_examples_to_generate = 16

    # load training and validation data
    ds_train, ds_val, ds_info = load_dataset(batch_size)
    print(ds_info)

    # initialize and compile model
    optimizer = tf.keras.optimizers.Adam(1e-4)
    model = CVAE(latent_dim)

    # Pick a sample of the test set for generating output images
    # assert batch_size >= num_examples_to_generate
    for test_batch in ds_val.take(1):
        test_sample = test_batch[0]
        generate_and_save_images(model, 0, test_sample)

    # Iterate over epochs.
    for epoch in range(1, epochs + 1):
        print("Start of epoch %d" % (epoch,))

        # Iterate over the batches of the dataset.
        start_time = time.time()
        for step, train_x in tqdm(enumerate(ds_train)):
            loss = train_step(model, train_x, optimizer)
            if step % 1 == 0:
                print("step %d: mean loss = %.4f" % (step, loss))
        end_time = time.time()

        # Calculate reconstruction error.
        loss = tf.keras.metrics.Mean()
        for test_x in ds_train:
            loss(compute_loss(model, test_x))
        elbo = -loss.result()
        print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'
              .format(epoch, elbo, end_time - start_time))

        # Output sample from current epoch.
        for test_batch in ds_val.take(1):
            test_sample = test_batch[0]
            generate_and_save_images(model, epoch, test_sample)
