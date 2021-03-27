import tensorflow as tf
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
from models.cvae import CVAE
from preprocessing import dataloader
from scipy.linalg import sqrtm
from tensorflow.keras.applications import InceptionV3


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


# calculate frechet inception distance
def compute_fid(model, images1, images2):
    # calculate activations
    act1 = model.predict(images1)
    act2 = model.predict(images2)
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


def generate_images(model, batch):
    batch_imgs, batch_imgs_masked, batch_labels = batch
    mean, logvar = model.encode(batch_imgs_masked)
    z = model.reparameterize(mean, logvar)
    predictions = model.sample(z)
    return predictions


def generate_and_save_images(step, epoch, batch, predictions, folder):
    batch_imgs, batch_imgs_masked, batch_labels = batch
    num_col = batch_imgs.shape[0]
    plt.figure(figsize=(num_col, 3))
    for i in range(num_col):
        fig = plt.subplot(3, num_col, i + 1)
        plt.imshow(batch_imgs[i, :, :, :])
        plt.axis('off')
        plt.subplot(3, num_col, i + 1 + num_col)
        plt.imshow(batch_imgs_masked[i, :, :, :])
        plt.axis('off')
        plt.subplot(3, num_col, i + 1 + num_col + num_col)
        plt.imshow(predictions[i, :, :, :])
        plt.axis('off')
        fig.set_title(int(batch_labels[i]))
    plt.savefig('vae/{}_out/step{:04d}_epoch{:04d}.png'.format(folder, step, epoch))
    plt.show()


if __name__ == "__main__":
    data_dir = 'coco2017'
    classes = ['airplane']
    model_dir = 'vae/'
    log_dir = 'logs'
    input_image_size = (256, 256, 3)
    num_steps = 6000
    batch_size = 10
    epochs = 2
    latent_dim = 32
    optimizer = tf.keras.optimizers.Adam(1e-3)

    # load and augment training data
    ds_train = dataloader(classes, data_dir, input_image_size, batch_size, 'val2017')
    ds_val = dataloader(classes, data_dir, input_image_size, batch_size, 'val2017')

    # Initialize and compile model
    model = CVAE(latent_dim, input_image_size)
    model_incept = InceptionV3(include_top=False, pooling='avg', input_shape=input_image_size)
    callback_list = [tf.keras.callbacks.TensorBoard(log_dir=log_dir)]
    TC = tf.keras.callbacks.CallbackList(callbacks=callback_list, model=model)

    # Set up logs for Tensorboard
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = log_dir + '/gradient_tape/' + current_time + '/train'
    test_log_dir = log_dir + '/gradient_tape/' + current_time + '/test'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)
    manager = tf.train.CheckpointManager(ckpt, "vae/checkpoints", max_to_keep=5)

    # Pick a sample of the test set for generating output images
    # assert batch_size >= num_examples_to_generate
    test_batch = next(ds_val)
    predictions = generate_images(model, test_batch)
    generate_and_save_images(0, 0, test_batch, predictions, 'train')

    # Iterate over epochs.
    for epoch in range(1, epochs + 1):
        print("Start of epoch %d" % (epoch,))

        # Iterate over the batches of the dataset.
        train_loss = tf.keras.metrics.Mean()
        val_loss = tf.keras.metrics.Mean()
        train_fid = tf.keras.metrics.Mean()
        val_fid = tf.keras.metrics.Mean()
        start_time = time.time()
        for step, (train_x, val_x) in enumerate(zip(ds_train, ds_val)):
            # Split batches into images and labels
            train_imgs, train_imgs_masked, train_labels = train_x
            val_imgs, val_imgs_masked, val_labels = val_x

            # Calculate reconstruction error on training and validation set
            train_loss(train_step(model, train_imgs_masked, optimizer))
            val_loss(compute_loss(model, val_imgs_masked))

            # Generate samples and compute FID
            train_pred = generate_images(model, train_x)
            val_pred = generate_images(model, val_x)
            train_fid(compute_fid(model_incept, train_imgs_masked, train_pred))
            val_fid(compute_fid(model_incept, val_imgs_masked, val_pred))

            # Log losses for Tensorboard viz
            ckpt.step.assign_add(1)
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', train_loss.result(), step=step)
                tf.summary.scalar('FID', train_fid.result(), step=step)
            with test_summary_writer.as_default():
                tf.summary.scalar('loss', val_loss.result(), step=step)
                tf.summary.scalar('FID', val_fid.result(), step=step)

            # Print progress and display images
            if step % 200 == 0:
                print("step %d: mean train loss = %.4f" % (step, train_loss.result()))
                print("step %d: mean val loss = %.4f" % (step, val_loss.result()))
                print("step %d: mean train FID = %.4f" % (step, train_fid.result()))
                print("step %d: mean val FID = %.4f" % (step, val_fid.result()))
                generate_and_save_images(step, epoch, test_batch, train_pred, 'train')
                save_path = manager.save()
                print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
            if step == num_steps:
                break

        end_time = time.time()

        elbo = -val_loss.result()
        print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'
              .format(epoch, elbo, end_time - start_time))

    TC.on_train_end('_')
