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
    num_anomalies = 5

    # Load and augment test data
    ds_test = dataloader(classes, data_dir, input_image_size, batch_size, mode)

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
    # Calculate reconstruction error for each example
    for step, test_x in enumerate(ds_test):
        img, img_masked, label = test_x
        step_loss = model.compute_loss(img_masked)
        losses.append((step_loss, test_x))
        if step == 500:
            break

    # Sort examples by losses and get top n anomalies
    losses = sorted(losses, key=lambda x: x[0], reverse=True)
    for step, (step_loss, test_x) in enumerate(losses):
        pred = generate_images(model, test_x)
        show_images(step, 0, test_x, pred, anomaly_dir)
        print("step %d: loss = %.4f" % (step, step_loss))
        if step > num_anomalies - 1:
            break
