import tensorflow as tf
import tensorflow_datasets as tfds
from labelencoder import LabelEncoder
from utils import preprocess_data


def load_dataset(batch_size):
    # Construct a tf.data.Dataset
    (train_dataset, val_dataset), ds_info = tfds.load(
        'coco/2017',
        split=['train', 'validation'],
        with_info=True,
        data_dir="data"
    )

    # Build training pipeline
    label_encoder = LabelEncoder()
    autotune = tf.data.experimental.AUTOTUNE
    train_dataset = train_dataset.map(preprocess_data, num_parallel_calls=autotune)
    train_dataset = train_dataset.shuffle(8 * batch_size)
    train_dataset = train_dataset.padded_batch(
        batch_size=batch_size, padding_values=(0.0, 1e-8, -1), drop_remainder=True
    )
    train_dataset = train_dataset.map(
        label_encoder.encode_batch, num_parallel_calls=autotune
    )
    train_dataset = train_dataset.apply(tf.data.experimental.ignore_errors())
    train_dataset = train_dataset.prefetch(autotune)

    # Build validation pipeline
    val_dataset = val_dataset.map(preprocess_data, num_parallel_calls=autotune)
    val_dataset = val_dataset.padded_batch(
        batch_size=1, padding_values=(0.0, 1e-8, -1), drop_remainder=True
    )
    val_dataset = val_dataset.map(label_encoder.encode_batch, num_parallel_calls=autotune)
    val_dataset = val_dataset.apply(tf.data.experimental.ignore_errors())
    val_dataset = val_dataset.prefetch(autotune)

    return train_dataset, val_dataset, ds_info
