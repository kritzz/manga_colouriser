import os
# supress tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import sys
# CHANGE DIRECTORY
sys.path.append('/dcs/21/u2146727/cs310/local/utility/')
from utility.preprocess import get_images_path, augment_test, load_image_test_user_input
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.data import Dataset
import tensorflow as tf
# get cwd


def process_colours(model):
  cwd = os.getcwd()
  input_location = os.path.join(cwd, 'static/upload/')
  test1 = get_images_path(input_location)
  test_Data = Dataset.from_tensor_slices((test1, test1))
  test_Data = test_Data.map(augment_test)
  test_Data = test_Data.batch(1)
  if model == "xdog":
    location = os.path.join(cwd, 'static/generator_edges.h5')
  elif model == "greyscale":
    location = os.path.join(cwd, 'static/generator_greyscale.h5')
  new_model = load_model(location, compile=False)
  for inp, tar in test_Data.take(1):
      colourise(new_model, inp, tar)
  return

def process_hints():
  cwd = os.getcwd()
  input_location = os.path.join(cwd, 'static/upload/')
  test1 = get_images_path(input_location)
  test_Data = Dataset.from_tensor_slices(test1)
  test_Data = test_Data.map(lambda x: tf.py_function(load_image_test_user_input, [x], [tf.float32, tf.float32])).shuffle(100)
  test_Data = test_Data.batch(1)
  location = os.path.join(cwd, 'static/generator-third.h5')
  new_model = load_model(location, compile=False)
  for inp, tar in test_Data.take(1):
    colourise(new_model, inp, tar)
  return

def colourise(model, test_input, tar):
    prediction = model(test_input, training=True)
    cwd = os.getcwd()
    if np.isnan(prediction[0]).any() or np.isinf(prediction[0]).any():
      print("Invalid values detected in prediction. Handling them...")
    img = (prediction[0] * 0.5 + 0.5) * 255
    img = Image.fromarray(np.uint8(img))
    input_location = os.path.join(cwd, 'static/colours/')
    img.save(input_location + "output.jpg")

# file should be in cwd/static

