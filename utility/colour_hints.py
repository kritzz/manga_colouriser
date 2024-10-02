from PIL import Image
import numpy as np
import cv2


# sample colour hints from coloured image
def create_colour_hints(image, num_samples=10, patch_size=(20, 20)):
    height, width, channels = image.shape
    # Initialize the matrix to store samples
    samples_matrix = np.zeros((height, width, 3), dtype=image.dtype)
    
    for _ in range(num_samples):
        start_row = np.random.randint(0, height - patch_size[0] + 1)
        start_col = np.random.randint(int(width * 0.25), int(width * 0.75) - patch_size[1] + 1)
        
        patch = image[start_row:start_row+patch_size[0], start_col:start_col+patch_size[1], :]
        # apply guassian blur to the patch
        patch = cv2.GaussianBlur(patch, (5,5), 0)
        # make the patch values be the average of the patch
        patch = np.mean(patch, axis=(0, 1)).astype(np.float32)

        samples_matrix[start_row:start_row+patch_size[0], start_col:start_col+patch_size[1], :] = patch
    
    return samples_matrix

# takes a line art an dapplies random colour hints to it
def create_colour_hints_random(image, num_samples=5, patch_size=(20, 20)):
    height, width, channels = image.shape
    # Initialize the matrix to store samples
    samples_matrix = np.zeros((height, width, 3), dtype=image.dtype)
    for _ in range(num_samples):
        start_row = np.random.randint(0, height - patch_size[0] + 1)
        start_col = np.random.randint(int(width * 0.25), int(width * 0.75) - patch_size[1] + 1)
        # set the wholr patch to a same random colour
        patch = np.random.randint(0, 256, 3).astype(np.uint8)
        samples_matrix[start_row:start_row+patch_size[0], start_col:start_col+patch_size[1], :] = patch
    
    return samples_matrix


# manually add colour hints to the image
def colour_hint_manual(current_mask, coordinates, colour):
    # make the 20x20 patch to the colour given
    # and add it to the current mask
    patch = np.array(colour).astype(np.uint8)
    for coord in coordinates:
        start_row = coord[0]
        start_col = coord[1]
        current_mask[start_row:start_row+20, start_col:start_col+20, :] = patch
    return current_mask

#  Example of how to add manual colour hitns

# image = Image.open('/dcs/large/u2146727/kaggle/data/val100/324109.png')
# split the image into left and right
# left, right = image.crop((0, 0, 512, 512)), image.crop((512, 0, 1024, 512))
# image = np.array(left)

# height, width, channels = image.shape
# samples = np.zeros((height, width, 3), dtype=image.dtype)
# samples = colour_hint_manual(samples, [(60, 256)], [71, 49, 49])
# samples = colour_hint_manual(samples, [(200, 180)], [71, 49, 49])
# # samples = colour_hint_manual(samples, [(400, 220)], [225, 0, 0])
# #samples = colour_hint_manual(samples, [(250, 300)], [230, 232, 220])
# samples = colour_hint_manual(samples, [(380, 370)], [222, 18, 18])
# samples = colour_hint_manual(samples, [(400, 250)], [222, 18, 18])
# samples = colour_hint_manual(samples, [(400, 150)], [222, 18, 18])
# samples = colour_hint_manual(samples, [(170, 260)], [235, 195, 96])


# mask = np.any(samples > 0, axis=2)
# result_image = np.where(mask[:, :, np.newaxis], samples, right)

# # Display the result
# plt.figure(figsize=(8, 6))
# plt.imshow(result_image)
# plt.show()
# # save result image under the same name as the original image
# # save the image
# result_image = Image.fromarray(result_image)
# result_image.save('/dcs/large/u2146727/kaggle/data/val_manual/324109.png')