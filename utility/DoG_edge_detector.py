import os
import numpy as np
import cv2
from skimage.color import rgb2gray
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_otsu
from tqdm import tqdm


LARGE_PATH = "/dcs/large/u2146727/edgecropped"
SMALL_PATH = "/dcs/21/u2146727/cs310/dataset/edgecropped"

# algorithm referenced from the github repo below
# Source : https://github.com/CemalUnal/XDoG-Filter
# Reference : XDoG: An eXtended difference-of-Gaussians compendium including advanced image stylization
# Link : http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.365.151&rep=rep1&type=pdf


# get all images path from dataset
def get_images_path(folder):
    images_path = []
    for filename in os.listdir(folder):
        images_path.append(os.path.join(folder, filename))
    return images_path


def extract_number_from_filename(filename):
    return int(''.join(filter(str.isdigit, filename)))



def DoG_edge_detector(im, tau=7, phi=5, eps=-0.1, k=5, sigma=0.6):
    imf1 = gaussian_filter(im, sigma)
    imf2 = gaussian_filter(im, sigma * k)
    imdiff = (1+tau)*imf1 - tau * imf2
    imdiff = (imdiff < eps) * 1.0 + (imdiff >= eps) * (1.0 + np.tanh(phi * imdiff))
    imdiff -= imdiff.min()
    imdiff /= imdiff.max()
    th = threshold_otsu(imdiff)
    imdiff = imdiff >= th
    # apply noise reduction median filter
    imdiff = cv2.medianBlur(imdiff.astype('float32'), 3)
    imdiff = (imdiff * 255).astype('float32')
    return imdiff




# use dog edge detector to get the edges of the the images in colour folder
# output_folder = os.path.join("/dcs/large/u2146727/edge")
# # get number of images in the folder
# os.makedirs(output_folder, exist_ok=True)


# for each image in the folder, apply the dog edge detector and save the black and white image and use a tdqm progress bar to show the progress
def apply_DoG_edge_detector(path, test=False ):
    if test:
        path = path[:10]
    try:
        for i in tqdm(range(len(path)), "applying DoG edge detector"):
            im = cv2.imread(path[i])
            original_number = extract_number_from_filename(os.path.basename(path[i]))
            if im.shape[2] == 3:
                im = rgb2gray(im)
            imdiff = DoG_edge_detector(im)
            # save the black and white image to the specified filename with the number at the end
            filename = os.path.join(LARGE_PATH, f'{original_number}.jpg')
            cv2.imwrite(filename, imdiff)
    except Exception as e:
        print(f"Error applying DoG edge detector: {str(e)}")

def apply_greyscale(path, test=False):
    if test:
        path = path[:10]
    try:
        for i in tqdm(range(len(path)), "applying greyscale"):
            im = cv2.imread(path[i])
            original_number = extract_number_from_filename(os.path.basename(path[i]))
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            filename = os.path.join("/dcs/21/u2146727/cs310/dataset/greyscale/", f'{original_number}.jpg')
            cv2.imwrite(filename, im)
    except Exception as e:
        print(f"Error applying greyscale: {str(e)}")

path = get_images_path("/dcs/21/u2146727/cs310/dataset/colourcropped/")
# apply_DoG_edge_detector(path)
apply_greyscale(path)