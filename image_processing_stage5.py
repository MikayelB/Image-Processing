import os
import cv2
import numpy as np

# Adjust this values to get something useful, so far I haven't gotten anything useful but in theory it shuold work (I think)
threshold_area = 100
aspect_ratio_min = 0.2
aspect_ratio_max = 2.0

def detect_and_label_characters(binary_regions):
    """
    Detect and label specific characters in binary regions.

    Some research findings:
    We can use geometric and statistical properties such as style, circularity, horizontal/vertical position of components, etc., to get specific characters.
    """

    # Finding contours
    contours, _ = cv2.findContours(binary_regions, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    labeled_image = np.zeros_like(binary_regions)

    for contour in contours:
        # Calculating contour area and bounding box properties
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)

        # Appling our labeling criteria based on geometric and statistical properties
        if area > threshold_area and aspect_ratio_min < w / h < aspect_ratio_max:
            cv2.putText(labeled_image, "Specific Character", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 2)

    return labeled_image

def process_image(input_path, output_folder):
    binary_image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

    # Detecting and labeling specific characters
    labeled_image = detect_and_label_characters(binary_image)

    # Saving the result image
    output_path = os.path.join(output_folder, os.path.basename(input_path).replace(".png", "_bin.png"))
    cv2.imwrite(output_path, labeled_image)

def main():
    for number in range(28, 33):
        input_folder = f"CSHandwriting/L0{number:02d}"
        output_folder = f"Stage5_Results/L0{number:02d}"

        os.makedirs(output_folder, exist_ok=True)

        for filename in os.listdir(input_folder):
            if filename.endswith(".png"):
                input_path = os.path.join(input_folder, filename)
                process_image(input_path, output_folder)

if __name__ == "__main__":
    main()
