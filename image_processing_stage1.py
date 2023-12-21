import os
import cv2
import numpy as np

# --------------STAGE 1--------------

def remove_printed_text(image):
    # Using thresholding
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    blurred = cv2.medianBlur(thresh, 5)
    return blurred

def remove_instructors_marks(image):
    # Using color filtering
    lower_red = np.array([0, 0, 100], dtype=np.uint8)
    upper_red = np.array([100, 100, 255], dtype=np.uint8)
    
    # Resize the boundaries to match the size of the input image
    lower_red = np.resize(lower_red, image.shape[:2] + (1,))
    upper_red = np.resize(upper_red, image.shape[:2] + (1,))
    
    mask = cv2.inRange(image, lower_red, upper_red)
    result = cv2.bitwise_and(image, image, mask=~mask)
    return result


def extract_handwriting(image):
    # Using edge detection
    edges = cv2.Canny(image, 50, 150)
    return edges

def crop_and_save(image, output_path):
    # Using bounding box detection
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        cropped = image[y:y+h, x:x+w]
        cv2.imwrite(output_path, cropped)

def adjust_brightness_contrast(image, alpha=1.5, beta=50, gamma=1.2):
    # Convert the image to grayscale if it has multiple channels
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.shape[-1] == 3 else image

    # Brightness and contrast adjustment
    adjusted = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)

    # Gamma correction
    adjusted_gamma = np.power(adjusted / 255.0, gamma) * 255.0

    # Merging the adjusted channel with the original image
    result = cv2.merge([adjusted_gamma.astype(np.uint8)] * 3) if image.shape[-1] == 3 else adjusted_gamma.astype(np.uint8)

    return result


def convert_to_binary(image):
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

def process_image(input_path, output_folder):
    image = cv2.imread(input_path)

    # Remove printed text
    no_text = remove_printed_text(image)

    # Remove instructor's marks
    no_marks = remove_instructors_marks(no_text)

    # Extract handwriting
    handwriting = extract_handwriting(no_marks)

    # Crop and save
    output_path = os.path.join(output_folder, os.path.basename(input_path).replace(".png", "_bin.png"))
    crop_and_save(handwriting, output_path)

    # Adjust brightness/contrast
    adjusted_image = adjust_brightness_contrast(handwriting)

    # Convert to binary format
    binary_image = convert_to_binary(adjusted_image)

    # Save the binary image
    binary_output_path = os.path.join(output_folder, os.path.basename(input_path).replace(".png", "_bin.png"))
    cv2.imwrite(binary_output_path, binary_image)

def main():
    for number in range(28, 33):
        input_folder = f"CSHandwriting/L0{number:02d}"
        output_folder = f"Stage1_Results/L0{number:02d}"

        for filename in os.listdir(input_folder):
            if filename.endswith(".png"):
                input_path = os.path.join(input_folder, filename)
                process_image(input_path, output_folder)

if __name__ == "__main__":
    main()
