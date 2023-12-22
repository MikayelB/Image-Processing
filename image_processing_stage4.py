import os
import cv2
import numpy as np

def construct_binary_regions(image):
    """
    Apply morphological operations to construct binary regions.

    Some research findings:
    Morphological operations (ex. dilation and erosion) can be used to construct binary regions
    in handwriting samples and the quality of binary regions is often impaced based on the choice of kernel size.
    """

    kernel = np.ones((5, 5), np.uint8)
    dilated_image = cv2.dilate(image, kernel, iterations=1)
    eroded_image = cv2.erode(dilated_image, kernel, iterations=1)

    return eroded_image

def separate_characters_and_keywords(binary_regions):
    """
    Implement methods to separate characters, keywords, etc.

    Some research findings:
    Contour detection and blob analysis can be used for this.
    """

    # Finding contours
    contours, _ = cv2.findContours(binary_regions, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw bounding boxes around detected objects
    result_image = cv2.cvtColor(binary_regions, cv2.COLOR_GRAY2BGR)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return result_image

def process_image(input_path, output_folder):
    binary_image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

    # Constructing binary regions
    binary_regions = construct_binary_regions(binary_image)

    # Separating characters, keywords, etc.
    separated_image = separate_characters_and_keywords(binary_regions)

    # Saving the result image
    output_path = os.path.join(output_folder, os.path.basename(input_path).replace(".png", "_bin.png"))
    cv2.imwrite(output_path, separated_image)


def main():
    for number in range(28, 33):
        input_folder = f"CSHandwriting/L0{number:02d}"
        output_folder = f"Stage4_Results/L0{number:02d}"

        os.makedirs(output_folder, exist_ok=True)

        for filename in os.listdir(input_folder):
            if filename.endswith(".png"):
                input_path = os.path.join(input_folder, filename)
                process_image(input_path, output_folder)

if __name__ == "__main__":
    main()
