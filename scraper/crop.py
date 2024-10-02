#python function that crops an image to 512x512 pixels
from PIL import Image
import os
from tqdm import tqdm


def get_images_path(folder):
    images_path = []
    for filename in os.listdir(folder):
        images_path.append(os.path.join(folder, filename))
    return images_path

def extract_number_from_filename(filename):
    return int(''.join(filter(str.isdigit, filename)))


# CHANGE DIRECTORY
path = "/dcs/large/u2146727/colour/"

# a testing directory, obsolete
test_path = "/dcs/21/u2146727/cs310/dataset/colour/"
testing = False
if testing:
    images_path = get_images_path(test_path)
else:
    images_path = get_images_path(path)

#crop the images to 512x512 pixels

def crop_image(image_path, coords):
    image_obj = Image.open(image_path)
    # if the image is RGBA or CMYK, convert it to RGB
    if image_obj.mode in ('RGBA', 'CMYK'):
        image_obj = image_obj.convert('RGB')
    cropped_image = image_obj.crop(coords)
    return cropped_image

def apply_crop_image(path, coords, test = False):
    for i in tqdm(range(len(path)), "cropping images"):
        cropped_image = crop_image(path[i], coords)
        original_number = extract_number_from_filename(os.path.basename(path[i]))
        if test:
            filename = os.path.join("/dcs/21/u2146727/cs310/dataset/test/", f'{original_number}.jpg')
        else:
            #CHANGE DIRECTORY to output folder
            filename = os.path.join("/dcs/large/u2146727/originalcropped", f'{original_number}.jpg')
        print (filename)
        # save the cropped image to the specified filename
        cropped_image.save(filename)


# CHANGE DIRECTORY
images_path= get_images_path("/dcs/large/u2146727/original/")
# apply_crop_image(images_path, (300, 600, 812, 1112))
apply_crop_image(images_path, (50, 512, 562, 1024))


