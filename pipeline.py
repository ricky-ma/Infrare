import tensorflow as tf
import tensorflow_datasets as tfds


def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label


def load_dataset():
    builder = tfds.builder(name="coco", data_dir="C:/Users/mrric/tensorflow_datasets")
    config = tfds.download.DownloadConfig(
        extract_dir="C:/Users/mrric/tensorflow_datasets/coco"
    )
    builder.download_and_prepare(download_config=config)

    # Construct a tf.data.Dataset
    (ds_train, ds_val), ds_info = tfds.load(
        'coco/2017',
        split=['train', 'validation'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
        download=False,
    )

    # Build training pipeline
    ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(128)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

    # Build validation pipeline
    ds_val = ds_val.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_val = ds_val.batch(128)
    ds_val = ds_val.cache()
    ds_val = ds_val.prefetch(tf.data.experimental.AUTOTUNE)

    # # Build evaluation pipeline
    # ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # ds_test = ds_test.batch(128)
    # ds_test = ds_test.cache()
    # ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

    return ds_train, ds_val
