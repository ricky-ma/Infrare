import tensorflow as tf
import datetime
from models.cvae import CVAE
from preprocessing import dataloader
from train import compute_loss, generate_and_save_images


if __name__ == "__main__":
    data_dir = 'coco2017'
    classes = ['orange']
    mode = 'val2019'
    model_dir = 'vae/'
    log_dir = 'logs'
    input_image_size = (256, 256, 3)
    batch_size = 1
    latent_dim = 32

    # Load and augment test data
    ds_test = dataloader(classes, data_dir, input_image_size, batch_size, mode)

    # Set up variables for Tensorboard
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    test_log_dir = log_dir + '/gradient_tape/' + current_time + '/test'
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    # Initialize model from checkpoint
    model = CVAE(latent_dim, input_image_size)
    optimizer = tf.keras.optimizers.Adam(1e-3)
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)
    manager = tf.train.CheckpointManager(ckpt, "vae/checkpoints", max_to_keep=5)
    ckpt.restore(manager.latest_checkpoint)

    losses = []
    labels = []

    # Calculate reconstruction error for each example
    loss = tf.keras.metrics.Mean()
    for step, test_x in enumerate(ds_test):
        img, label = test_x
        step_loss = compute_loss(model, img)
        loss(step_loss)
        with test_summary_writer.as_default():
            tf.summary.scalar('loss', loss.result(), step=step)
        print("step %d: loss = %.4f" % (step, step_loss))

        # Store losses and labels of each example
        losses.append(step_loss)
        labels.append(label)

        # Outlier/anomaly if loss for example is 2x mean loss
        if step_loss > 2*loss.result():
            generate_and_save_images(model, step, test_x, 'anomaly')

        if step == 200:
            break
