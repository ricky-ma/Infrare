import tensorflow as tf
import time
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
from models.cvae import CVAE
from preprocessing import dataloader


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


def generate_and_save_images(model, epoch, batch, train):
    batch_imgs, batch_labels = batch
    mean, logvar = model.encode(batch_imgs)
    z = model.reparameterize(mean, logvar)
    predictions = model.sample(z)
    num_col = batch_imgs.shape[0]
    fig = plt.figure(figsize=(num_col, 2))

    for i in range(num_col):
        fig = plt.subplot(2, num_col, i + 1)
        plt.imshow(batch_imgs[i, :, :, :])
        plt.axis('off')

        plt.subplot(2, num_col, i + 1 + num_col)
        plt.imshow(predictions[i, :, :, :])
        plt.axis('off')

        fig.set_title(int(batch_labels[i]))

    if train:
        plt.savefig('vae/train_out/image_at_epoch_{:04d}.png'.format(epoch))
    else:
        plt.savefig('vae/test_out/image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()


if __name__ == "__main__":
    data_dir = 'coco2017'
    classes = ['orange']
    mode = 'val2017'
    model_dir = 'vae/'
    log_dir = 'logs'
    input_image_size = (512, 512, 3)
    batch_size = 10
    epochs = 2
    latent_dim = 32
    optimizer = tf.keras.optimizers.Adam(1e-3)

    # load and augment training data
    ds_train = dataloader(classes, data_dir, input_image_size, batch_size, mode)

    # initialize and compile model
    model = CVAE(latent_dim, input_image_size)
    callback_list = [
        tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    ]
    TC = tf.keras.callbacks.CallbackList(callbacks=callback_list, model=model)
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = log_dir + '/gradient_tape/' + current_time + '/train'
    test_log_dir = log_dir + '/gradient_tape/' + current_time + '/test'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    # Pick a sample of the test set for generating output images
    # assert batch_size >= num_examples_to_generate
    test_batch = next(ds_train)
    generate_and_save_images(model, 0, test_batch, True)

    # Iterate over epochs.
    for epoch in range(1, epochs + 1):
        print("Start of epoch %d" % (epoch,))

        # Iterate over the batches of the dataset.
        start_time = time.time()
        for step, train_x in enumerate(ds_train):
            imgs, labels = train_x
            loss = train_step(model, imgs, optimizer)
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', loss, step=step)
            if step % 100 == 0:
                print("step %d: mean loss = %.4f" % (step, loss))
                generate_and_save_images(model, step, test_batch, True)
                checkpoint_dir = os.path.dirname('vae/checkpoints/cp-{step:04d}')
                model.save_weights(checkpoint_dir)
        end_time = time.time()

        # Calculate reconstruction error.
        loss = tf.keras.metrics.Mean()
        for step, test_x in enumerate(ds_train):
            imgs, labels = test_x
            loss(compute_loss(model, test_x))
            with test_summary_writer.as_default():
                tf.summary.scalar('loss', loss, step=step)
        elbo = -loss.result()
        print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'
              .format(epoch, elbo, end_time - start_time))

        # # Output sample from current epoch.
        # test_batch = next(ds_train)[0]
        # generate_and_save_images(model, epoch, test_batch)
    TC.on_train_end('_')
