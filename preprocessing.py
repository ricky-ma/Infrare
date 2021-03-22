from pycocotools.coco import COCO
import numpy as np
import random
import skimage.io as io
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def get_class_name(class_id, cats):
    for i in range(len(cats)):
        if cats[i]['id'] == class_id:
            return cats[i]['name']
    return None


def filter_dataset(folder, classes=None, mode='train'):
    # initialize COCO api for instance annotations
    ann_file = '{}/annotations/instances_{}.json'.format(folder, mode)
    coco = COCO(ann_file)
    images = []
    if classes is not None:
        # iterate for each individual class in the list
        for className in classes:
            # get all images containing given categories
            cat_ids = coco.getCatIds(catNms=className)
            img_ids = coco.getImgIds(catIds=cat_ids)
            images += coco.loadImgs(img_ids)
    else:
        img_ids = coco.getImgIds()
        images = coco.loadImgs(img_ids)

    # Now, filter out the repeated images
    unique_images = []
    for i in range(len(images)):
        if images[i] not in unique_images:
            unique_images.append(images[i])

    random.shuffle(unique_images)
    dataset_size = len(unique_images)
    return unique_images, dataset_size, coco


def get_image(image_obj, img_folder, input_image_size):
    # Read and normalize an image
    train_img = io.imread(img_folder + '/' + image_obj['file_name']) / 255.0
    # Resize
    # TODO: fix cropping to center objects in image
    train_img = cv2.resize(train_img, input_image_size)
    if len(train_img.shape) == 3 and train_img.shape[2] == 3:  # If it is a RGB 3 channel image
        return train_img
    else:  # To handle a black and white image, increase dimensions to 3
        stacked_img = np.stack((train_img,) * 3, axis=-1)
        return stacked_img


def get_binary_masks(image_obj, coco, cat_ids, input_image_size):
    masks = []
    labels = []
    ann_ids = coco.getAnnIds(image_obj['id'], catIds=cat_ids, iscrowd=None)
    for ann_id in ann_ids:
        anns = coco.loadAnns(ann_id)
        mask = np.zeros(input_image_size)

        for a in range(len(anns)):
            new_mask = cv2.resize(coco.annToMask(anns[a]), input_image_size)
            # Threshold because resizing may cause extraneous values
            new_mask[new_mask >= 0.5] = 1
            new_mask[new_mask < 0.5] = 0
            mask = np.maximum(new_mask, mask)

        # Add extra dimension for parity with train_img size [X * X * 3]
        mask = mask.reshape(input_image_size[0], input_image_size[1])
        masks.append(mask)
        labels.append(anns[0]['category_id'])
    return masks, labels


def augment_data(gen, aug_generator_args, seed=None):
    # Initialize the image data generator with args provided
    image_gen = ImageDataGenerator(**aug_generator_args)
    # Remove the brightness argument for the mask. Spatial arguments similar to image.
    aug_generator_args_mask = aug_generator_args.copy()
    _ = aug_generator_args_mask.pop('brightness_range', None)
    np.random.seed(seed if seed is not None else np.random.choice(range(9999)))

    for img, label in gen:
        seed = np.random.choice(range(9999))
        # keep the seeds syncronized otherwise the augmentation of the images
        # will end up different from the augmentation of the masks
        g_x = image_gen.flow(255 * img,
                             batch_size=img.shape[0],
                             seed=seed,
                             shuffle=True)

        img_aug = next(g_x) / 255.0
        yield img_aug, label


def data_generator(images, classes, coco, folder, input_image_size=(224, 224, 3),
                   batch_size=4, mode='train'):
    img_folder = '{}/images/{}'.format(folder, mode)
    dataset_size = len(images)
    cat_ids = coco.getCatIds(catNms=classes)
    input_image_size = input_image_size[:2]

    c = 0
    while True:
        img = np.zeros((batch_size, input_image_size[0], input_image_size[1], 3)).astype('float')
        lab = np.empty(batch_size)
        for i in range(c, c + batch_size):  # initially from 0 to batch_size, when c = 0
            image_obj = images[i]
            # Retrieve image
            train_img = get_image(image_obj, img_folder, input_image_size)
            # Mask image
            masks, labels = get_binary_masks(image_obj, coco, cat_ids, input_image_size)
            for mask, label in zip(masks, labels):
                mask = mask[:, :, np.newaxis]
                train_img = train_img * mask
                # Add to respective batch sized arrays
                img[i - c] = train_img
                lab[i - c] = label

        c += batch_size
        if c + batch_size >= dataset_size:
            c = 0
            random.shuffle(images)
        yield img, lab


def dataloader(classes, data_dir, input_image_size, batch_size, mode):
    images, dataset_size, coco = filter_dataset(data_dir, classes, mode)
    data_gen = data_generator(images, classes, coco, data_dir, input_image_size, batch_size, mode)
    aug_generator_args = dict(featurewise_center=False,
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
    dataset = augment_data(data_gen, aug_generator_args)
    return dataset
