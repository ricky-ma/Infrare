import tensorflow as tf
import time
import numpy as np
import matplotlib.pyplot as plt
from model import CVAE
from preprocessing import filter_dataset, dataloader, augment_data


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
        plt.imshow(predictions[i, :, :, 0])
        plt.axis('off')

    # tight_layout minimizes the overlap between 2 sub-plots
    plt.savefig('vae/results/image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()


if __name__ == "__main__":
    data_dir = 'coco2017'
    classes = ['laptop', 'tv', 'cell phone']
    mode = 'val2017'
    model_dir = "vae/"
    input_image_size = (512, 512, 3)
    mask_type = 'normal'
    batch_size = 4
    epochs = 10
    latent_dim = 2  # set the dimensionality of the latent space to a plane for visualization later
    num_examples_to_generate = 16

    # load and augment training data
    images, dataset_size, coco = filter_dataset(data_dir, classes, mode)
    train_gen = dataloader(images, classes, coco, data_dir, input_image_size, batch_size, mode, mask_type)
    augGeneratorArgs = dict(featurewise_center=False,
                            samplewise_center=False,
                            rotation_range=5,
                            width_shift_range=0.01,
                            height_shift_range=0.01,
                            brightness_range=(0.8, 1.2),
                            shear_range=0.01,
                            zoom_range=[1, 1.25],
                            horizontal_flip=True,
                            vertical_flip=False,
                            fill_mode='reflect',
                            data_format='channels_last')
    ds_train = augment_data(train_gen, augGeneratorArgs)

    # initialize and compile model
    optimizer = tf.keras.optimizers.Adam(1e-3)
    model = CVAE(latent_dim, input_image_size)

    # Pick a sample of the test set for generating output images
    # assert batch_size >= num_examples_to_generate
    test_batch = next(ds_train)[0]
    generate_and_save_images(model, 0, test_batch)

    # Iterate over epochs.
    for epoch in range(1, epochs + 1):
        print("Start of epoch %d" % (epoch,))

        # Iterate over the batches of the dataset.
        start_time = time.time()
        for step, train_x in enumerate(ds_train):
            # Only use image, labels are not necessary
            train_x = train_x[0]
            loss = train_step(model, train_x, optimizer)
            if step % 100 == 0:
                print("step %d: mean loss = %.4f" % (step, loss))
                generate_and_save_images(model, step, test_batch)
        end_time = time.time()

        # Calculate reconstruction error.
        loss = tf.keras.metrics.Mean()
        for test_x in ds_train:
            test_x = test_x[0]
            loss(compute_loss(model, test_x))
        elbo = -loss.result()
        print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'
              .format(epoch, elbo, end_time - start_time))

        # # Output sample from current epoch.
        # test_batch = next(ds_train)[0]
        # generate_and_save_images(model, epoch, test_batch)
