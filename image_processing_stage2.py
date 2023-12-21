import os
import cv2
import numpy as np
import pytesseract

def remove_printed_text(image):
    # Using pytesseract for OCR
    text = pytesseract.image_to_string(image)
    print("OCR Text:\n", text)

    # Assuming OCR has detected the text regions, I create a mask to remove them
    mask = np.zeros_like(image)
    cv2.putText(mask, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY_INV)

    # Apply the mask to remove text
    no_text = cv2.bitwise_and(image, image, mask=mask)
    return no_text

def evaluate_page_features(image):
    # Edge detection for orientation evaluation
    edges = cv2.Canny(image, 50, 150)
    # Below if uncommented will give a popup of an image
    # cv2.imshow("Edges", edges)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def process_image(input_path, output_folder):
    image = cv2.imread(input_path)

    # Remove printed text
    no_text = remove_printed_text(image)

    # Evaluate page features
    evaluate_page_features(no_text)

    # Save the processed image
    output_path = os.path.join(output_folder, os.path.basename(input_path).replace(".png", "_bin.png"))
    cv2.imwrite(output_path, no_text)

def main():
    # Seting the Tesseract executable path for Linux, uncomment below for Windows
    # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

    for number in range(28, 33):
        input_folder = f"CSHandwriting/L0{number:02d}"
        output_folder = f"Stage2_Results/L0{number:02d}"

        os.makedirs(output_folder, exist_ok=True)

        for filename in os.listdir(input_folder):
            if filename.endswith(".png"):
                input_path = os.path.join(input_folder, filename)
                process_image(input_path, output_folder)

if __name__ == "__main__":
    main()
