import os
import cv2
import numpy as np

"""
For a hope of some extra credit I came up with this stage.
Stage 6: Image Binarization with Adaptive Thresholding
"""

def apply_adaptive_threshold(image):
    """
    Apply adaptive thresholding to convert the image into a binary format.

    Some research findings:
    We can use adaptive thresholding to handle variations in lighting and contrast(useful to distinguish
    features in handwriting samples)
    """

    # Applying adaptive thresholding
    binary_image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    return binary_image

def process_image(input_path, output_folder):
    # Reading the image in grayscale
    grayscale_image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

    # Applying adaptive thresholding
    binary_image = apply_adaptive_threshold(grayscale_image)

    # Saving the result image
    output_path = os.path.join(output_folder, os.path.basename(input_path).replace(".png", "_bin.png"))
    cv2.imwrite(output_path, binary_image)

def main():
    for number in range(28, 33):
        input_folder = f"CSHandwriting/L0{number:02d}"
        output_folder = f"Stage6_Results/L0{number:02d}"

        os.makedirs(output_folder, exist_ok=True)

        for filename in os.listdir(input_folder):
            if filename.endswith(".png"):
                input_path = os.path.join(input_folder, filename)
                process_image(input_path, output_folder)
if __name__ == "__main__":
    main()
