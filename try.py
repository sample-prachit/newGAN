import cv2
import numpy as np

# Read the two images
image1 = cv2.imread('/Users/prachit/self/Working/OCT/generative/deepfake_gi_fastGAN/custom_dataset/images/image1.png')  # Replace with your first image path
image2 = cv2.imread('/Users/prachit/self/Working/OCT/generative/deepfake_gi_fastGAN/custom_dataset/masks/mask1.png')  # Replace with your second image path

# Check if images are loaded properly
if image1 is not None and image2 is not None:
    # Print shapes of both images
    print(f"Shape of image1: {image1.shape}")
    print(f"Shape of image2: {image2.shape}")
else:
    print("Error: Could not load one or both images")