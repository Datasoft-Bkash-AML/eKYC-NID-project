import pytesseract
from PIL import Image
import cv2
import numpy as np

config = '--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789'

# Load image
pil_image = Image.open('history/6e62cbac-876d-4af1-97ab-95f8e42b507b/FrontSide/nationalId_TC2.jpg').convert("RGB")
image = np.array(pil_image)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# Apply Gaussian blur
blurred = cv2.GaussianBlur(gray, (3, 3), 0)

# Apply adaptive thresholding
binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 11, 2)

# OCR
text = pytesseract.image_to_string(binary, lang='eng', config=config)
print(text)
