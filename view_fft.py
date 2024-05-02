import cv2 as cv
import numpy as np
import os

# Path to the folder containing images
folder_path = 'small_sticker_goal_vid'

# Get list of image files in the folder
image_files = sorted([os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(('.png', '.jpg', '.jpeg'))])

if not image_files:
    print("No image files found in the folder.")
    exit()

# Load the first image to get dimensions
first_image = cv.imread(image_files[0], 0)
height, width = first_image.shape

for image_file in image_files:
    # Read the image
    image = cv.imread(image_file, 0)

    # Perform FFT
    f_transform = np.fft.fft2(image)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = 20 * np.log(1 + np.abs(f_shift))
    magnitude_spectrum = np.uint8(255 * magnitude_spectrum / np.max(magnitude_spectrum))
    concatenated_image = np.hstack((magnitude_spectrum, image))

    cv.namedWindow('FFT Image', cv.WINDOW_NORMAL)
    cv.resizeWindow('FFT Image', 1200, 720)

    cv.imshow('FFT Image', concatenated_image)
    cv.waitKey(0)

cv.destroyAllWindows()
