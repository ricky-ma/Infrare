import datetime
import time
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy.linalg import sqrtm
from tensorflow.keras.applications import InceptionV3
from models import VAE, CVAE, VPGA, MemCAE, AE, CAE
from preprocessing import dataloader


@tf.function
def train_step(model, x, optimizer):
    """
    Executes one training step and returns the loss.
    This function computes the loss and gradients, and uses the latter to
    update the model's parameters.
    """
    if model.architecture == "VPGA":
        with tf.GradientTape(persistent=True) as tape:
            enc_loss, dec_loss = model.compute_loss(x)
            loss = enc_loss + dec_loss
        enc_grads = tape.gradient(enc_loss, model.trainable_variables)
        dec_grads = tape.gradient(dec_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(enc_grads + dec_grads, model.trainable_variables))
        del tape
    else:
        with tf.GradientTape() as tape:
            loss = model.compute_loss(x)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


# calculate frechet inception distance
def compute_fid(incept_model, images1, images2):
    # calculate activations
    act1 = incept_model.predict(images1)
    act2 = incept_model.predict(images2)
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
    if model.architecture == "MemCAE":
        z, c = model.encode(batch_imgs_masked)
    elif model.architecture in ("AE", "CAE"):
        z = model.encode(batch_imgs_masked)
    else:
        mean, logvar = model.encode(batch_imgs_masked)
        z = model.reparameterize(mean, logvar)
    predictions = model.sample(z)
    return predictions


def show_images(step, epoch, batch, predictions, folder):
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
    plt.savefig('output/{}_out/step{:04d}_epoch{:04d}.png'.format(folder, step, epoch))
    plt.show()


if __name__ == "__main__":
    data_dir = 'data/coco2017'
    classes = ['orange']
    model_dir = 'output/'
    log_dir = 'logs'
    input_image_size = (256, 256, 3)
    num_steps = 6000
    batch_size = 10
    epochs = 1
    latent_dim = 32
    adam_optimizer = tf.keras.optimizers.Adam(1e-3)

    # load and augment training data
    ds_train = dataloader(classes, data_dir, input_image_size, batch_size, 'train2019')
    ds_val = dataloader(classes, data_dir, input_image_size, batch_size, 'val2019')

    # Initialize and compile model
    # vae_model = AE(latent_dim, input_image_size)
    # vae_model = CAE(latent_dim, input_image_size)
    # vae_model = VAE(latent_dim, input_image_size)
    vae_model = CVAE(latent_dim, input_image_size)
    # vae_model = MemCAE(latent_dim, True, input_image_size, batch_size)
    model_incept = InceptionV3(include_top=False, pooling='avg', input_shape=input_image_size)
    callback_list = [tf.keras.callbacks.TensorBoard(log_dir=log_dir)]
    TC = tf.keras.callbacks.CallbackList(callbacks=callback_list, model=vae_model)

    # Set up logs for Tensorboard
    train_log_dir = log_dir + '/gradient_tape/' + vae_model.architecture + '/multi/train'
    test_log_dir = log_dir + '/gradient_tape/' + vae_model.architecture + '/multi/test'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=adam_optimizer, net=vae_model)
    manager = tf.train.CheckpointManager(ckpt, "output/checkpoints", max_to_keep=5)

    # Pick a sample of the test set for generating output images
    # assert batch_size >= num_examples_to_generate
    test_batch = next(ds_val)
    preds = generate_images(vae_model, test_batch)

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
            train_loss(train_step(vae_model, train_imgs_masked, adam_optimizer))
            val_loss(vae_model.compute_loss(val_imgs_masked))

            # Log losses for Tensorboard viz
            ckpt.step.assign_add(1)
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', train_loss.result(), step=step)
            with test_summary_writer.as_default():
                tf.summary.scalar('loss', val_loss.result(), step=step)

            # Print progress and display images
            if step % 100 == 0:
                # Generate samples and compute FID
                test_pred = generate_images(vae_model, test_batch)
                train_pred = generate_images(vae_model, train_x)
                val_pred = generate_images(vae_model, val_x)
                train_fid(compute_fid(model_incept, train_imgs_masked, train_pred))
                val_fid(compute_fid(model_incept, val_imgs_masked, val_pred))
                with train_summary_writer.as_default():
                    tf.summary.scalar('FID', train_fid.result(), step=step)
                with test_summary_writer.as_default():
                    tf.summary.scalar('FID', val_fid.result(), step=step)

                print("step %d: mean train loss = %.4f" % (step, train_loss.result()))
                print("step %d: mean val loss = %.4f" % (step, val_loss.result()))
                print("step %d: mean train FID = %.4f" % (step, train_fid.result()))
                print("step %d: mean val FID = %.4f" % (step, val_fid.result()))
                show_images(step, epoch, test_batch, test_pred, 'train')
                show_images(step, epoch, val_x, val_pred, 'val')
                save_path = manager.save()
                print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
            if step == num_steps:
                break

        end_time = time.time()

        elbo = -val_loss.result()
        print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'
              .format(epoch, elbo, end_time - start_time))

    TC.on_train_end('_')
