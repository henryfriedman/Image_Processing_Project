import cv2 as cv
import os
import scipy

from skimage import io, color, data, draw, exposure, feature, filters, measure, morphology, util, segmentation
import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as sft
import scipy.signal as sps
from PIL import Image

def get_dft_magnitudes(dft):
    dfts = scipy.fft.fftshift(dft)
    mag = np.abs(dfts)
    lmag = np.log10(1 + mag)
    return lmag

sticker = np.asarray(Image.open('triangle.png'))

# Path to the folder containing images
triangles_path = 'triangles_vid'

# Get list of image files in the folder
image_files = sorted([os.path.join(triangles_path, file) for file in os.listdir(triangles_path) if file.endswith(('.png', '.jpg', '.jpeg'))])

if not image_files:
    print("No image files found in the folder.")
    exit()

# Load the first image to get dimensions
first_image = cv.imread(image_files[0], 0)
height, width = first_image.shape

for image_file in image_files:
    image = cv.imread(image_file, 0)
    box_height = sticker.shape[0]
    box_width = sticker.shape[1]

    # Create an empty composite array
    composite_dft = np.zeros_like(padded_image1)

    total_height = composite_dft.shape[0]
    total_width = composite_dft.shape[1]

    num_boxes_vertically = total_height//box_height
    num_boxes_horizontally = total_width//box_width

    # Populate the composite array with DFT results
    for i in range(num_boxes_vertically):
        for j in range(num_boxes_horizontally):
            # Calculate the index in the list
            idx = i * num_boxes_horizontally + j
            # Place the DFT result in the correct position in the composite array
            composite_dft[
                i * box_height: (i + 1) * box_height,
                j * box_width: (j + 1) * box_width
            ] = get_dft_magnitude(dft1[idx])

    combined_image = np.concatenate((image, composite_dft), axis=1)
    cv.imshow('fft', combined_image)
    cv.waitKey(30)

cv.destroyAllWindows()
