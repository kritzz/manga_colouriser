import os
# supress tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
import random
from PIL import Image
import numpy as np
import cv2

# input a folder path, return a list of images path
def get_images_path(folder):
    images_path = []
    for filename in os.listdir(folder):
        images_path.append(os.path.join(folder, filename))
    return images_path


# input an image, and return the image tensor
def get_images(black_white_path, colour_path):
    try:
        bw_image = tf.io.read_file(black_white_path)
        colour_image = tf.io.read_file(colour_path)
        # decode the image
        bw_image = tf.image.decode_image(bw_image, channels=3)
        colour_image = tf.image.decode_image(colour_image, channels=3)
        bw_image.set_shape([None, None, 3])
        colour_image.set_shape([None, None, 3])
        return bw_image, colour_image
    except Exception as e:
        print("Error in get_images:", e)
        print("Black and white path:", black_white_path)
        print("Colour path:", colour_path)

# data augmentation
def crop_mirror(bw_image_original, colour_image_original, height=486, width=486, crop = True):
    try:
        bw_image_resize = tf.image.resize(bw_image_original, [height, width],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        colour_image_resize = tf.image.resize(colour_image_original, [height, width],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        # crop to 256x256
        stack = tf.stack([bw_image_resize, colour_image_resize])
        if crop:
            stack = tf.image.random_crop(stack, size=[2, 256, 256, 3])
        # mirror with 50% chance
        random_number = random.random()
        if random_number > 0.5:
            stack = tf.image.flip_left_right(stack)
        bw, colour = tf.unstack(stack, axis=0)
        return bw, colour
    except Exception as e:
        print("Error processing tensor:", e)
        print("Black and white image:", bw_image_original)
        print("Colour image:", colour_image_original)

# normalize the pixel values between -1 and 1
def normalize(first, second):
    try:
        first = tf.dtypes.cast(first, tf.float32)
        second = tf.dtypes.cast(second, tf.float32)
        # Normalize the pixel values between -1 and 1
        first = (first / (255.0 / 2.0)) - 1.0
        second = (second / (255.0 / 2.0)) - 1.0

        return first, second
    except Exception as e:
        print("Error in normalise")

# pipeline for data augmentation
def augment_data(black_path, colour_path):
    black, colour = get_images(black_path, colour_path)
    black, colour = crop_mirror(black, colour)
    black, colour = normalize(black, colour)
    return black, colour

# pipeline for test data
def augment_test(black_path, colour_path):
    black, colour = get_images(black_path, colour_path)
    black = tf.image.resize(black, [256, 256],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    colour = tf.image.resize(colour, [256, 256],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    black, colour = normalize(black, colour)
    return black, colour

# colour hints 
def load(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_png(image)

    w = tf.shape(image)[1]

    w = w // 2
    real_image = image[:, :w, :]
    input_image = image[:, w:, :]

    input_image = tf.cast(input_image, tf.float32)
    real_image = tf.cast(real_image, tf.float32)

    return input_image, real_image




# colour hints training
def load_image_train(image_file):
    input_image, real_image = load(image_file)
    input_image = create_color_hints(input_image, real_image, num_samples=10)
    input_image, real_image = crop_mirror(input_image, real_image, height = 256, width = 256, crop = False)
    input_image, real_image = normalize(input_image, real_image)
    return input_image, real_image


# colour hint testing
def load_image_test(image_file):
    input_image, real_image = load(image_file)
    input_image = create_color_hints(input_image, real_image, num_samples=10)
    input_image = tf.image.resize(input_image, [256, 256],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tf.image.resize(real_image, [256, 256],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    input_image, real_image = normalize(input_image, real_image)

    return input_image, real_image

# colour hint testing with random colour hints
def load_image_test_random(image_file):
    input_image, real_image = load(image_file)
    input_image = create_color_hints_random(input_image, real_image, num_samples=5)
    input_image = tf.image.resize(input_image, [256, 256],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tf.image.resize(real_image, [256, 256],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    input_image, real_image = normalize(input_image, real_image)

    return input_image, real_image

# for manual colour hints
def load_single(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_png(image)
    image = tf.cast(image, tf.float32)
    return image

# colour hint testing with user input 
def load_image_test_user_input(image_file):
    input_image= load_single(image_file)
    input_image = tf.image.resize(input_image, [256, 256],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = input_image
    input_image, real_image = normalize(input_image, real_image)
    return input_image, real_image


# create colour hints from original image
def create_color_hints(lineart, image, num_samples=10, patch_size=(20, 20)):
    height = 512
    width = 512
    channels = 3
    samples_matrix = np.zeros((height, width, 3), dtype=np.float32)
    image = np.array(image)
    
    for _ in range(num_samples):
        start_row = np.random.randint(0, height - patch_size[0] + 1)
        start_col = np.random.randint(int(width * 0.25), int(width * 0.75) - patch_size[1] + 1)
        
        patch = image[start_row:start_row+patch_size[0], start_col:start_col+patch_size[1], :]
        # apply guassian blur to the patch
        patch = cv2.GaussianBlur(patch, (5,5), 0)
        # make the patch values be the average of the patch
        patch = np.mean(patch, axis=(0, 1)).astype(np.float32)

        samples_matrix[start_row:start_row+patch_size[0], start_col:start_col+patch_size[1], :] = patch
    
    mask = np.any(samples_matrix > 0, axis=2)
    result_image = np.where(mask[:, :, np.newaxis], samples_matrix, lineart)
    return result_image


# create colour hints randomly using line art 
def create_color_hints_random(lineart, image, num_samples=10, patch_size=(20, 20)):
    height = 512
    width = 512
    channels = 3
    samples_matrix = np.zeros((height, width, 3), dtype=np.float32)
    image = np.array(image)
    
    for _ in range(num_samples):
        start_row = np.random.randint(0, height - patch_size[0] + 1)
        start_col = np.random.randint(int(width * 0.25), int(width * 0.75) - patch_size[1] + 1)
        # set the whole patch to the same random color
        patch = np.random.randint(0, 256, 3).astype(np.float32)
        samples_matrix[start_row:start_row+patch_size[0], start_col:start_col+patch_size[1], :] = patch
    
    mask = np.any(samples_matrix > 0, axis=2)
    result_image = np.where(mask[:, :, np.newaxis], samples_matrix, lineart)
    return result_image