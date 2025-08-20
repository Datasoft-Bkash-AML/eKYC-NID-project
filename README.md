## 4. Bengali Language Data (ben.traineddata)

If you need to process Bengali text, you must install the Bengali language data for Tesseract.

### Ubuntu
```
sudo apt install tesseract-ocr-ben
```

### Windows
1. Download `ben.traineddata` from: https://github.com/tesseract-ocr/tessdata
2. Copy `ben.traineddata` to your Tesseract `tessdata` directory (e.g., `C:\Program Files\Tesseract-OCR\tessdata`).

If you get errors about missing language data, make sure the file is present in the correct directory.
# OCR Engine Installation Guide

## 1. Install Tesseract OCR

### Ubuntu
```
sudo apt update
sudo apt install tesseract-ocr libtesseract-dev
```

### Windows
1. Download the Tesseract installer from: https://github.com/tesseract-ocr/tesseract
2. Run the installer and follow the instructions.
3. Add the Tesseract installation path (e.g., `C:\Program Files\Tesseract-OCR`) to your Windows PATH environment variable.

## 2. Install pytesseract Python package

Run the following command in your project directory:
```
pip install pytesseract
```

## 3. Verify Installation

Run this in Python:
```python
import pytesseract
print(pytesseract.get_tesseract_version())
```

If you see a version number, the installation is successful.

## Troubleshooting
- If you get errors about Tesseract not found, make sure it is installed and the path is set correctly.
- On Windows, ensure the Tesseract folder is in your PATH.
- On Ubuntu, `tesseract` should be available in `/usr/bin/tesseract`.