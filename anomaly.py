import tensorflow as tf
import datetime
from models.cvae import CVAE
from preprocessing import dataloader
from train import compute_loss, generate_and_save_images


if __name__ == "__main__":
    data_dir = 'coco2017'
    classes = ['banana']
    mode = 'val2017'
    model_dir = 'vae/'
    log_dir = 'logs'
    input_image_size = (512, 512, 3)
    mask_img = True
    batch_size = 1
    latent_dim = 32
    latest = tf.train.latest_checkpoint('vae')

    # load and augment test data
    ds_test = dataloader(classes, data_dir, input_image_size, batch_size, mode)

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    test_log_dir = log_dir + '/gradient_tape/' + current_time + '/test'
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    model = CVAE(latent_dim, input_image_size)
    model.load_weights(latest)

    # Calculate reconstruction error.
    losses = []
    labels = []
    loss = tf.keras.metrics.Mean()
    for step, test_x in enumerate(ds_test):
        img, label = test_x
        loss(compute_loss(model, img))
        with test_summary_writer.as_default():
            tf.summary.scalar('loss', loss.result(), step=step)
        print("step %d: mean loss = %.4f" % (step, loss.result()))
        losses.append(loss.result())
        labels.append(label)
        generate_and_save_images(model, step, test_x, False)
