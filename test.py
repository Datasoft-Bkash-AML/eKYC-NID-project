import cv2
import numpy as np
from PIL import Image
import pytesseract
from pytesseract import image_to_data, image_to_string
import re
import os
import glob
from collections import Counter

def extract_nid_improved(image_path):
    """
    Improved NID extraction with multiple preprocessing approaches
    """
    try:
        # Load image
        pil_image = Image.open(image_path).convert("RGB")
        image = np.array(pil_image)
        
        # Multiple preprocessing approaches
        results = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply morphological operations to clean up the image
        kernel = np.ones((1,1), np.uint8)
        gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
        
        # Try different thresholding methods
        thresholding_methods = [
            ("Adaptive", cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)),
            ("OTSU", cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]),
            ("Simple", cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)[1])
        ]
        
        # Try different OCR configurations
        ocr_configs = [
            ("Whitelist digits", '--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789 '),
            ("PSM 8 digits", '--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789 '),
            ("PSM 7 digits", '--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789 '),
            ("Default PSM 6", '--oem 1 --psm 6'),
            ("Single word", '--oem 1 --psm 8'),
        ]
        
        # Test all combinations
        for thresh_name, binary in thresholding_methods:
            for config_name, config in ocr_configs:
                try:
                    # Using image_to_string
                    text = image_to_string(binary, lang='eng', config=config).strip()
                    if text:
                        results.append(f"{thresh_name}+{config_name}: {text}")
                        
                except Exception as e:
                    continue
        
        # Extract and clean potential NID numbers
        nid_candidates = []
        
        # Improved regex patterns for NID detection
        nid_patterns = [
            r'\b\d{3}\s*\d{3}\s*\d{4}\b',  # 190 015 9060 format
            r'\b\d{10}\b',                   # 1900159060 format
            r'\b\d{2,3}[-\s]*\d{3}[-\s]*\d{4}\b',  # flexible format
        ]
        
        print(f"\n🔍 OCR Results for {os.path.basename(image_path)}:")
        for result in results:
            print(f"  📄 {result}")
            
            # Clean the text
            cleaned_text = re.sub(r'[^\d\s-]', '', result)
            
            # Try to find NID patterns
            for pattern in nid_patterns:
                matches = re.findall(pattern, cleaned_text)
                for match in matches:
                    # Clean the match (remove spaces and hyphens)
                    cleaned_match = re.sub(r'[-\s]+', '', match)
                    if len(cleaned_match) == 10 and cleaned_match.isdigit():
                        nid_candidates.append(cleaned_match)
                        print(f"  ✅ Found NID candidate: {cleaned_match}")
        
        # If no direct pattern match, try to extract all digits
        if not nid_candidates:
            all_digits_str = ''.join(re.findall(r'\d', ' '.join(results)))
            print(f"  🔢 All extracted digits: {all_digits_str}")
            
            # Look for 10-digit sequences
            if len(all_digits_str) >= 10:
                for i in range(len(all_digits_str) - 9):
                    candidate = all_digits_str[i:i+10]
                    if len(candidate) == 10:
                        nid_candidates.append(candidate)
                        print(f"  💡 10-digit sequence found: {candidate}")
        
        # Return the most likely candidate
        if nid_candidates:
            # Return the most common candidate
            most_common = Counter(nid_candidates).most_common(1)
            best_candidate = most_common[0][0] if most_common else nid_candidates[0]
            print(f"  🎯 Best candidate: {best_candidate}")
            return best_candidate
        
        print(f"  ❌ No NID found")
        return None
        
    except Exception as e:
        print(f"  ❌ Error processing {image_path}: {str(e)}")
        return None

def extract_nid_region_based(image_path):
    """
    Alternative approach: Focus on the region where NID is typically located
    """
    try:
        pil_image = Image.open(image_path).convert("RGB")
        image = np.array(pil_image)
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Focus on the region where "NID No" text appears
        height, width = gray.shape
        # Crop to focus on upper portion where NID number typically appears
        roi = gray[int(height*0.1):int(height*0.7), int(width*0.1):int(width*0.9)]
        
        # Enhance the ROI
        roi = cv2.resize(roi, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        
        # Apply bilateral filter to reduce noise while preserving edges
        roi = cv2.bilateralFilter(roi, 9, 75, 75)
        
        # Apply OTSU thresholding
        _, roi_binary = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # OCR with specific configuration for numbers
        config = '--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789 '
        text = image_to_string(roi_binary, lang='eng', config=config)
        
        print(f"  🎯 ROI OCR result: {text}")
        
        # Extract 10-digit numbers
        digits = re.findall(r'\d', text)
        if len(digits) >= 10:
            all_digits = ''.join(digits)
            # Look for sequences that might be NID (starting with 1 or 2 typically for Bangladesh)
            for i in range(len(all_digits) - 9):
                candidate = all_digits[i:i+10]
                if candidate.startswith(('1', '2')):  # Bangladesh NID typically starts with 1 or 2
                    return candidate
            
            # If no good candidate, return first 10 digits
            return all_digits[:10] if len(all_digits) >= 10 else None
        
        return None
        
    except Exception as e:
        print(f"  ❌ ROI Error: {str(e)}")
        return None

def find_image_files():
    """
    Find all image files in common locations
    """
    # Common image extensions
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    
    # Common locations to search
    search_paths = [
        '.',  # Current directory
        'images/',
        'test_images/',
        'history/**/FrontSide/',  # Your original path structure
    ]
    
    image_files = []
    
    for search_path in search_paths:
        for ext in extensions:
            pattern = os.path.join(search_path, ext)
            files = glob.glob(pattern, recursive=True)
            image_files.extend(files)
    
    return list(set(image_files))  # Remove duplicates

def test_nid_extraction():
    """
    Test NID extraction on available images
    """
    print("🔍 Searching for image files...")
    
    # First, try to find image files automatically
    image_files = find_image_files()
    
    if not image_files:
        print("❌ No image files found automatically.")
        print("Please ensure your images are in one of these locations:")
        print("  - Current directory")
        print("  - images/ folder")
        print("  - test_images/ folder")
        print("\nOr provide the correct path to your images.")
        
        # Ask user for image paths
        print("\n💡 You can also specify image paths manually:")
        manual_paths = input("Enter image paths separated by commas (or press Enter to skip): ").strip()
        
        if manual_paths:
            image_files = [path.strip() for path in manual_paths.split(',')]
        else:
            return
    
    print(f"📁 Found {len(image_files)} image file(s):")
    for i, img_path in enumerate(image_files):
        print(f"  {i}: {img_path}")
    
    print("\n" + "="*60)
    
    # Test each image
    for i, image_path in enumerate(image_files):
        if not os.path.exists(image_path):
            print(f"\n--- Image {i}: {os.path.basename(image_path)} ---")
            print(f"❌ File not found: {image_path}")
            continue
            
        print(f"\n--- Image {i}: {os.path.basename(image_path)} ---")
        
        # Try improved method
        print("🚀 Method 1: Improved OCR")
        nid1 = extract_nid_improved(image_path)
        
        # Try region-based method
        print("\n🎯 Method 2: Region-based OCR")
        nid2 = extract_nid_region_based(image_path)
        
        print(f"\n📊 Final Results:")
        print(f"  Method 1 (Improved): {nid1 if nid1 else 'None'}")
        print(f"  Method 2 (Region): {nid2 if nid2 else 'None'}")
        
        # Determine best result
        if nid1 and nid2 and nid1 == nid2:
            print(f"  🎉 Consistent result: {nid1}")
        elif nid1:
            print(f"  ✅ Best guess: {nid1} (from Method 1)")
        elif nid2:
            print(f"  ✅ Best guess: {nid2} (from Method 2)")
        else:
            print(f"  ❌ No NID number extracted")
        
        print("-" * 60)

if __name__ == "__main__":
    test_nid_extraction()