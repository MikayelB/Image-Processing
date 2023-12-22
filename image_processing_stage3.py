import os
import cv2
import numpy as np

def detect_straight_lines(image):
    # Hough Transform to detect straight lines
    lines = cv2.HoughLines(image, 1, np.pi / 180, threshold=100)

    # Creating a blank image to draw lines (with the same color map as the original image)
    lines_image = np.zeros_like(image)

    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            # Drawing the line on the lines_image
            cv2.line(lines_image, (x1, y1), (x2, y2), 255, 2)

    return lines_image

def process_image(input_path, output_folder):
    binary_image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

    # Detecting straight lines in handwriting samples
    lines_image = detect_straight_lines(binary_image)

    # Saving the image with detected lines
    output_path = os.path.join(output_folder, os.path.basename(input_path).replace(".png", "_bin.png"))
    cv2.imwrite(output_path, lines_image)

def main():
    for number in range(28, 33):
        input_folder = f"CSHandwriting/L0{number:02d}"
        output_folder = f"Stage3_Results/L0{number:02d}"

        os.makedirs(output_folder, exist_ok=True)

        for filename in os.listdir(input_folder):
            if filename.endswith(".png"):
                input_path = os.path.join(input_folder, filename)
                process_image(input_path, output_folder)

if __name__ == "__main__":
    main()
