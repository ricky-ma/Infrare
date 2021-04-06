import datetime
import tensorflow as tf
from models import VAE, CVAE, VPGA, MemCAE
from preprocessing import dataloader
from train import show_images, generate_images

if __name__ == "__main__":
    data_dir = 'data/coco2017'
    classes = ['cat']
    mode = 'val2019'
    log_dir = 'logs'
    anomaly_dir = 'MemCAE/cat/anomaly'
    ckpt_dir = 'output/MemCAE/cat/checkpoints'
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
    optimizer = tf.keras.optimizers.Adam(1e-3)
    model = MemCAE(latent_dim, False, input_image_size, batch_size, 500, optimizer)
    if model.architecture == 'MemCAE':
        ckpt = model.get_ckpt()
    else:
        ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)
    manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=5)
    ckpt.restore(manager.latest_checkpoint)

    losses = []
    labels = []
    # Calculate reconstruction error for each example
    loss = tf.keras.metrics.Mean()
    for step, test_x in enumerate(ds_test):
        img, img_masked, label = test_x
        step_loss = model.compute_loss(img_masked)
        loss(step_loss)
        with test_summary_writer.as_default():
            tf.summary.scalar('loss', loss.result(), step=step)

        # Store losses and labels of each example
        losses.append(step_loss)
        labels.append(label)
        if step == 100:
            break

    # Outlier/anomaly if loss for example is 2x mean loss
    threshold = 1.05 * sum(losses) / len(losses)
    print("Threshold={}".format(threshold))
    for step, (step_loss, test_x) in enumerate(zip(losses, ds_test)):
        if step_loss > threshold:
            pred = generate_images(model, test_x)
            show_images(step, 0, test_x, pred, anomaly_dir)
            print("step %d: loss = %.4f" % (step, step_loss))
