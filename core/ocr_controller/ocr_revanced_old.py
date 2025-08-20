import cv2
import imutils
import numpy as np
import pytesseract
from PIL import ImageFont, ImageDraw, Image
import os
import re
import asyncio
import time
from datetime import datetime
from .face_matcher_controller import ArcFaceMatcher  # Assuming you have this library

faceMatcher = ArcFaceMatcher(threshold=80)

class OCRProcessor:
    def __init__(self):
        current_folder = os.path.dirname(os.path.abspath(__file__))  # Get the current script's directory
        parent_folder = os.path.dirname(current_folder)  # Go one level up
        target_folder = os.path.join(parent_folder, "assets")
        self.font_path = os.path.join(target_folder, 'ben.ttf')

    async def estimate_image_quality(self, image_path: str) -> dict:
        image = cv2.imread(image_path)
        if image is None:
            return {"message": "Failed to load image"}

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        height, width = gray.shape[:2]

        # --- Sharpness (Laplacian) ---
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

        # --- Brightness & Contrast ---
        brightness = np.mean(gray)
        contrast = np.std(gray)

        # --- Text-Like Region Detection (MSER) ---
        mser = cv2.MSER_create()
        regions, _ = mser.detectRegions(gray)
        text_like_regions = len(regions)

        # --- Glare Detection (overexposed white regions) ---
        overexposed = cv2.inRange(gray, 245, 255)
        glare_ratio = np.sum(overexposed > 0) / (height * width)

        # --- Skew Detection (Hough transform angle analysis) ---
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 150)
        skew_angle = 0
        if lines is not None:
            angles = [theta for rho, theta in lines[:, 0]]
            degrees = [abs(np.rad2deg(theta) - 90) for theta in angles]
            skew_angle = np.median(degrees) if degrees else 0

        # --- Scoring System ---
        sharpness_score = 1 if sharpness > 1000 else 0.3
        contrast_score = 1 if contrast > 50 else 0.5
        brightness_score = 1 if 60 < brightness < 150 else 0.3
        text_region_score = 1 if text_like_regions > 1000 else 0.5
        glare_score = 1 if glare_ratio < 0.05 else 0.2
        skew_score = 1 if skew_angle < 5 else 0.4

        score = (
            sharpness_score +
            contrast_score +
            brightness_score +
            text_region_score +
            glare_score +
            skew_score
        ) / 6 * 100

        if score >= 90:
            message = "Image is excellent for OCR"
        elif score >= 75:
            message = "Image is acceptable for OCR"
        else:
            message = "Image may cause issues with OCR"

        return {
            "sharpness": round(sharpness, 2),
            "contrast": round(contrast, 2),
            "brightness": round(brightness, 2),
            "text_like_regions": text_like_regions,
            "glare_ratio": round(glare_ratio, 4),
            "skew_angle_deg": round(skew_angle, 2),
            "score": round(score, 2),
            "message": message,
            "acceptableQuality": True if score >= 75 else False,
        }

    async def letsStart(self, image_path, unique_id, image_side, profile_path, show_steps=True):
        start_time = time.time()

        result = await self.processFrontSide(image_path, unique_id, profile_path, show_steps)
        end_time = time.time() 
        total_time = round(end_time - start_time, 2)
        result['total_time_in_seconds'] = total_time
        return result 
    
    async def processFrontSide(self, image_path, unique_id, profile_path, show_steps=True):
        # Preprocess the image
        nidImage = await asyncio.to_thread(cv2.imread, image_path)
        profileImage = await asyncio.to_thread(cv2.imread, profile_path)
        

        if nidImage is None:
            raise FileNotFoundError(f"Image not found at {image_path}")

        resizedImage = await asyncio.to_thread(imutils.resize, nidImage, height=600)
        
        faceMatchingResult = await faceMatcher.compare(profileImage,resizedImage)

        # faceMatchingResult = {
        #         "is_face_matched": False,
        #         "face_matched_score": 0.0,
        #         "face_matched_threshold": 0.0,
        #         "reason": "Face not detected",
        #         "face1": None,
        #         "face2": None,
        #         "face1_rect": None,
        #         "face2_rect": None
        #     }

        resizedCopyImage = resizedImage.copy()

        gray = await asyncio.to_thread(cv2.cvtColor, resizedCopyImage, cv2.COLOR_BGR2GRAY)

        # Face detection
        faceRectangles = (0, 0, 0, 0)
        
        if faceMatchingResult.get("face2_rect") is not None:
            x, y, w, h = faceMatchingResult.get("face2_rect")

            # Add extra space to include more context around the face
            extra_space = int(w * 0.15)

            x_new = (x - extra_space)
            w_new = (w + extra_space)
            y_new = (y - extra_space)
            h_new = (h + extra_space)

            faceRectangles = (x_new, y_new, w_new, h_new)
            face = resizedImage[y_new:h_new, x_new:w_new]
            face_height, _ = face.shape[:2]

            combined_img = await self.process_images_into_one(face, [faceMatchingResult.get("face1"),faceMatchingResult.get("face2")])  
            combined_img = await self.write_text_to_image(
                combined_img, extra_space, 0, 0, 0, f"Result (Matched-{faceMatchingResult['is_face_matched']})"
            )
            combined_img = await self.write_text_to_image(
                combined_img, extra_space, face_height-extra_space, 0, 0, f"(Score-{faceMatchingResult['face_matched_score']}) (TScore-{faceMatchingResult['face_matched_threshold']})"
            )
            output_path = os.path.join(f"./history/{unique_id}/FrontSide", f"FrontSide.Faces.jpg")
            await asyncio.to_thread(cv2.imwrite, output_path, combined_img)

        # Enhance contrast and brightness
        alpha = 1.7
        beta = -60
        enhanced = await asyncio.to_thread(cv2.convertScaleAbs, gray, alpha=alpha, beta=beta)

        # Morphological operations
        rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5))
        gray = await asyncio.to_thread(cv2.GaussianBlur, gray, (3, 3), 0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(19, 19))
        grayNew = await asyncio.to_thread(clahe.apply, gray)
        _, grayNew = await asyncio.to_thread(
            cv2.threshold, grayNew, 165, 255, cv2.THRESH_TRUNC + cv2.THRESH_OTSU
        )

        blackhatNew = await asyncio.to_thread(
            cv2.morphologyEx, grayNew, cv2.MORPH_BLACKHAT, rectKernel
        )
        gradX = await asyncio.to_thread(cv2.Sobel, blackhatNew, cv2.CV_32F, 1, 0, -1)
        gradX = np.absolute(gradX)
        (minVal, maxVal) = (np.min(gradX), np.max(gradX))
        gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")

        # Thresholding and contour detection
        gradXNew = cv2.convertScaleAbs(gradX)
        largeKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
        gradXNew = await asyncio.to_thread(
            cv2.morphologyEx, gradX, cv2.MORPH_CLOSE, largeKernel
        )

        #---------------------------------------------------------
        _, thresh = await asyncio.to_thread(
            cv2.threshold, gradXNew, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
        )

        sqKernelNew = await asyncio.to_thread(cv2.getStructuringElement, cv2.MORPH_RECT, (5, 5))
        threshNew = await asyncio.to_thread(cv2.morphologyEx, thresh, cv2.MORPH_CLOSE, sqKernelNew)
        threshNew = await asyncio.to_thread(cv2.morphologyEx, thresh, cv2.MORPH_OPEN, sqKernelNew)
        
        contours, _ = await asyncio.to_thread(
            cv2.findContours, threshNew, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        tasks = []
        for contour in contours:
            area = await asyncio.to_thread(cv2.contourArea, contour)

            if area < 1000:  # Apply your threshold condition
                # Asynchronously draw each contour on the image
                tasks.append(asyncio.to_thread(cv2.drawContours, threshNew, [contour], -1, (0, 0, 0), thickness=cv2.FILLED))

        # Run all contour-drawing tasks concurrently
        await asyncio.gather(*tasks)

        # Define the square kernel asynchronously
        sqKernel = await asyncio.to_thread(cv2.getStructuringElement, cv2.MORPH_RECT, (19, 19))

        # Perform the closing operation asynchronously
        close = await asyncio.to_thread(cv2.morphologyEx, threshNew, cv2.MORPH_CLOSE, sqKernel)

        # Perform erosion asynchronously
        close = await asyncio.to_thread(cv2.erode, close, None, iterations=4)

        # Perform dilation asynchronously
        close = await asyncio.to_thread(cv2.dilate, close, None, iterations=6)

        contours, _ = await asyncio.to_thread(cv2.findContours, close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])

        # Start OCR processing
        detectImages = [enhanced, gray, grayNew, resizedCopyImage]
        gray_finalImage,ocrResults = await self.start_ocr_frontSide(resizedImage, detectImages, contours, faceRectangles, show_steps, unique_id)

        for key, value in ocrResults.items():
            if(key=='dob' and value):
                # Valid dates: 01 JAN 2024, 31 DEC 2023
                # dob_pattern = r'O?\d{1,2}\s[A-Za-z]{3}\s\d{4}'
                dob_pattern = r'O?\d{1,2}\s[A-Za-z]{3}\s(?:\d\s?){4}'

                def clean_date(date_str):
                    parts = date_str.split()
                    if len(parts) == 4:  # e.g. ['01', 'JAN', '2', '024']
                        parts[2] = parts[2] + parts[3]
                        parts = parts[:3]
                    elif len(parts) == 5:  # e.g. ['01', 'JAN', '2', '0', '24']
                        parts[2] = parts[2] + parts[3] + parts[4]
                        parts = parts[:3]
                    return " ".join(parts)

                def validate_date(date_str):
                    # Check if the format matches "DD MMM YYYY"
                    if re.match(dob_pattern, date_str):
                        try:
                            # Try to parse the date and check if it's valid
                            datetime.strptime(date_str, "%d %b %Y")
                            return True
                        except ValueError:
                            return False  # Date is invalid
                    else:
                        return False  # Format is incorrect
                # Define patterns

                benTexts = re.search(dob_pattern, value, re.IGNORECASE)
                value = benTexts.group(0) if benTexts else value
                value = clean_date(value)
                value = value if validate_date(value) else None 
                ocrResults[key] = value
                # Append OCR results

            elif(key=='nid_num' and value):
                # Valid numbers: 123 456 7890, 123-456-7890, 1234567890

                # Define patterns
                ten_digits_pattern = r'\b(\d{3}[-\s]?\d{3}[-\s]?\d{4})\b'

                benTexts = re.search(ten_digits_pattern, value, re.IGNORECASE)
                value = benTexts.group(0) if benTexts else value

                # Remove all characters except
                value = re.sub(r'(?<!\w)T(?!\w)', '', value)
                value = re.sub(r'[^0-9T]', '', value)

                # Replace T with 7
                ocrResults[key] = value.replace('T', '7')
            
            # ["bng_name", "eng_name", "father_name", "mother_name", "dob", "nid_num"]
            elif(key=='eng_name' and value):
                # Define patterns
                # Step 1: Remove Bengali prefix if present
                value = re.sub(r'^\s*Name\s*[:：]?\s*', '', value)

                # Step 2: Keep only English letters (both cases), dots, and spaces
                value = re.sub(r'[^a-zA-Z\s\.]', '', value)

                # Normalize multiple spaces
                value = re.sub(r'\s+', ' ', value)

                # Assign cleaned value to OCR results
                ocrResults[key] = value.strip()
                
            elif(key=='father_name' and value):
                # Define patterns
                # Step 1: Remove Bengali prefix if present
                value = re.sub(r'^\s*পিতা\s*[:：]?\s*', '', value)

                # Optional: Further clean up — keeping only Bengali letters, dots, and spaces
                value = re.sub(r'[^\u0980-\u09FF\s\.]', '', value)

                # Assign cleaned value to OCR results
                ocrResults[key] = value.strip()

            elif(key=='mother_name' and value):
                # Define patterns
                # Step 1: Remove Bengali prefix if present
                value = re.sub(r'^\s*মাতা\s*[:：]?\s*', '', value)

                # Optional: Further clean up — keeping only Bengali letters, dots, and spaces
                value = re.sub(r'[^\u0980-\u09FF\s\.]', '', value)

                # Assign cleaned value to OCR results
                ocrResults[key] = value.strip()
                
            elif value:
                ocrResults[key] = re.sub(r'\d', '', value)

            # Remove leading and trailing spaces
            ocrResults[key] = ocrResults[key].strip() if ocrResults[key] else ocrResults[key]

        # if faceDetected:
        #     _, buffer = cv2.imencode('.jpg', face)
        #     base64_image = base64.b64encode(buffer).decode('utf-8')
        #     ocrResults['nid_face'] = base64_image
        
        if show_steps:
            # Draw rectangle around the largest face asynchronously
            (x_f, y_f, w_f, h_f) = faceRectangles
            await asyncio.to_thread(
                cv2.rectangle, gray_finalImage, (x_f, y_f), (w_f, h_f), (0, 255, 0), 2
            )
            # detectImages = detectImages + [close, gray_finalImage, face]
            detectImages = detectImages + [close, gray_finalImage]
            
            combined_img = await self.process_images_into_one(resizedImage, detectImages)  
            output_path = os.path.join(f"./history/{unique_id}/FrontSide", f"FrontSide.Processed.jpg")
            await asyncio.to_thread(cv2.imwrite, output_path, combined_img)
        
        ocrResults['is_face_matched'] = faceMatchingResult['is_face_matched']
        ocrResults['face_matched_score'] = faceMatchingResult['face_matched_score']
        ocrResults['face_matched_threshold'] = faceMatchingResult['face_matched_threshold']
        
        return ocrResults

    async def start_ocr_frontSide(self, mainImage, detectImages, contours, faceRectangles, show_steps, unique_id):

            image_side = "FrontSide"
            x1, y1, x2, y2 = faceRectangles
            avgH, validContour = 0, 0
            first_contour, last_contour, max_width = None, None, 0
            
            valid_contours = []
            img_height, img_width = mainImage.shape[:2]
            # Tring to get all valid possible contours based on contour ratio
            for index, contour in enumerate(contours):
                x, y, w, h = cv2.boundingRect(contour)
    
                # Y-coordinate of the bottom edge of the bounding box
                bottom_of_contour = y + h  

                # Ignore contours that extend too close to the bottom edge of the image
                if bottom_of_contour >= img_height - (img_height*0.005):
                    continue  # Skip this contour
                
                aspect_ratio = w / float(h)
                area = cv2.contourArea(contour)
                
                if 1.5 < aspect_ratio and 1200 < area < 30000 and (x2-(x2/4)) < x and (y1-h)<y:
                    # print(f"Countour {index} values - x: {x}, y: {y}, w: {w}, h: {h}, area: {area} -> Accepted")
                    
                    if w > max_width:
                        max_width = w
                        
                    avgH += h
                    validContour += 1
                    valid_contours.append(contour)
                    # Set first contour if not already set
                    if first_contour is None:
                        first_contour = (x, y, w, h)

                    # Always update the last contour
                    last_contour = (x, y, w, h)
                
                # else:
                #     print(f"Countour {index} values - x: {x}, y: {y}, w: {w}, h: {h}, area: {area} -> Rejected")
                
            line_positions = {}
            
            # Calculate height/distance between the first and last contours and create possible text zone
            if first_contour and last_contour:
                first_y = first_contour[1]
                last_y = last_contour[1]
                last_h = last_contour[3]

                # Calculate the distance
                distance = abs((last_y + last_h) - (first_y))

                step_size = distance // 5

                # Define the meaningful names for lines
                names = ["bng_name", "eng_name", "father_name", "mother_name", "dob", "nid_num"]
                
                x_line = int(mainImage.shape[1]/1.6)
                for i in range(0, 6):
                    y_position = first_y + i * step_size
                    if show_steps:
                        cv2.line(mainImage, (x_line, y_position), (mainImage.shape[1], y_position), (0, 0, 255), 1)
                        cv2.putText(
                            mainImage,f"{names[i]}-{y_position}",(mainImage.shape[1]-150, y_position - 5),cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,  # Font scale
                            (0, 0, 255),  # Font color (green)
                            1,  # Thickness
                            cv2.LINE_AA  # Line type
                        )
                    line_positions[names[i]] = y_position


            avgH = avgH / validContour if validContour > 0 else 0

            mainImage = await self.write_text_to_image(
                mainImage, 10, 30, 0, 0, f"Details (avgH-{avgH}) (validContour-{validContour})"
            )

            ocrResults = []
            validContour = 0

            for i, contour in enumerate(valid_contours):
                x, y, w, h = cv2.boundingRect(contour)
                
                w = max_width

                if h > (avgH * 1.6):
                    h = h // 2
                    mainImage, ocrResult,detectContour = await self.process_text(mainImage, detectImages, x , y, w, h, validContour,unique_id,image_side, show_steps)
                    if ocrResult:
                        ocrResults.append((validContour, ocrResult, detectContour))
                    validContour += 1

                    mainImage, ocrResult, detectContour = await self.process_text(mainImage, detectImages, x , y + h , w , h, validContour,unique_id,image_side, show_steps)
                    if ocrResult:
                        ocrResults.append((validContour, ocrResult, detectContour))
                    validContour += 1
                else:
                    mainImage, ocrResult, detectContour = await self.process_text(mainImage, detectImages, x , y , w , h, validContour,unique_id,image_side,show_steps)
                    if ocrResult:
                        ocrResults.append((validContour, ocrResult, detectContour))
                    validContour += 1
                    
            intersections = await self.find_intersections_async(line_positions, ocrResults)

            return mainImage , intersections

    async def process_images_into_one(self, image, detectImages):
        processed_images = []

        # Process and add each detection image
        for detect_img in detectImages:
            # Resize to match original image height
            resized = await asyncio.to_thread(
                cv2.resize, 
                detect_img, 
                (image.shape[1], image.shape[0])
            )
            
            # Convert to BGR if grayscale
            if len(resized.shape) == 2:
                resized = await asyncio.to_thread(
                    cv2.cvtColor, 
                    resized, 
                    cv2.COLOR_GRAY2BGR
                )
            
            processed_images.append(resized)

        # Create rows with 2 images each
        rows = []
        for i in range(0, len(processed_images), 2):
            row_images = processed_images[i:i+2]
            if len(row_images) == 2:
                row = await asyncio.to_thread(cv2.hconcat, row_images)
                rows.append(row)
            elif len(row_images) == 1:
                # Add blank image to make pair if odd number
                blank = np.zeros_like(row_images[0])
                row = await asyncio.to_thread(cv2.hconcat, [row_images[0], blank])
                rows.append(row)

        # Combine all rows vertically
        final_combined = await asyncio.to_thread(cv2.vconcat, rows)
        
        return final_combined 

    async def find_intersections_async(self, line_positions, ocrResults):
        intersections = {key: [] for key in line_positions.keys()}
        
        for line_name, y_pos in line_positions.items():
            for contour in ocrResults:
                _, ocr_text, (x, y, w, h) = contour
                if y <= y_pos <= y + h or y-5 <= y_pos <= y + h:
                    intersections[line_name].append((ocr_text, x))
        
        # Merge OCR results if there are more than one for a line
        for line_name, ocr_texts in intersections.items():
            if len(ocr_texts) > 1:
                # Sort by x position
                ocr_texts.sort(key=lambda item: item[1])
                # Merge texts
                merged_text = ' '.join([text for text, _ in ocr_texts])
                intersections[line_name] = merged_text
            else:
                intersections[line_name] = ocr_texts[0][0] if ocr_texts else None

        # Remove special characters from all values and ensure spaces are not at the beginning or end
        for key in intersections:
            if intersections[key]:
                # Remove special characters but allow spaces within the text
                cleaned_value = re.sub(r'[^a-zA-Z0-9\u0980-\u09FF ]', '', intersections[key])
                # Remove spaces from the beginning and end
                cleaned_value = cleaned_value.strip()
                intersections[key] = cleaned_value

        return intersections
    
    async def image_sharpener(self, image):
        # Convert to grayscale
        # Ensure the image is 2D (grayscale)
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Enhance contrast
        alpha, beta = 1.7, -10
        enhanced = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)

        # Apply Gaussian blur
        blur = cv2.GaussianBlur(enhanced, (3, 3), 0)

        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY, 15, 2)

        # Sharpening filter
        kernel = np.array([[ 0, -1,  0], 
                        [-1,  5, -1], 
                        [ 0, -1,  0]])

        # Apply sharpening
        sharpened = cv2.filter2D(binary, -1, kernel)

        return sharpened.astype(np.uint8)
    
    def max_bangla_text(self,text_dict):
        bangla_pattern = re.compile(r'[\u0980-\u09FF]')  # Bangla Unicode range

        max_text = ""
        max_bangla_count = 0

        for key, text in text_dict.items():
            bangla_count = len(bangla_pattern.findall(text))  # Count Bangla characters
            if bangla_count > max_bangla_count:
                max_bangla_count = bangla_count
                max_text = text

        return max_text
    
    async def process_text(self, finalImage, detectImages, x, y, w, h, validContourNo,unique_id,image_side,show_steps, lang="ben+eng", ignoreFilter=False):
        
        extra_space_x = 20
        extra_space_y = 10
        icount = 0
        ocredtext = ''

        while icount < len(detectImages):
            detectImage = detectImages[icount]
                    
            x_new = max(x - extra_space_x, 0)
            w_new = min(w + 2 * extra_space_x, detectImage.shape[1] - x_new)
            y_new = max(y - extra_space_y, 0)
            h_new = min(h + 2 * extra_space_y, detectImage.shape[0] - y_new)
            
            # Crop the region of interest asynchronously
            cropped = detectImage[y_new:y_new + h_new, x_new:x_new + w_new]
            
            if show_steps:
                await asyncio.to_thread(cv2.imwrite, os.path.join(f"./history/{unique_id}/{image_side}", f"No{validContourNo}_TC{icount}.jpg"), cropped)

            # Perform OCR asynchronously
            ocredtext = await asyncio.to_thread(
                pytesseract.image_to_string, cropped, lang
            )
            ocredtext = ocredtext.strip()
            ocredtext = ocredtext.strip().replace("\n", "")
            if not ignoreFilter:
                ocredtext = ocredtext.replace("\x0c", "")
                
                # Only split if there's a colon and the part before it is less than 4 chars
                if ":" in ocredtext:
                    first_part, *rest = ocredtext.split(":", 1)
                    if len(first_part.strip()) < 4 :
                        ocredtext = rest[0].strip() if rest else ocredtext


            if len(ocredtext)>3:
                break
            else:
                icount += 1

            # Tried last time once by sharpening the image if no text detected
            if (icount == len(detectImages)) and len(ocredtext)<4:
                sharped = await self.image_sharpener(cropped)
                if show_steps:
                    await asyncio.to_thread(cv2.imwrite, os.path.join(f"./history/{unique_id}/{image_side}", f"No{validContourNo}_TC{icount+1}.jpg"), sharped)

                # Perform OCR asynchronously
                ocredtext = await asyncio.to_thread(
                    pytesseract.image_to_string, sharped, lang
                )
                ocredtext = ocredtext.strip()
                ocredtext = ocredtext.replace("\n", "")
                if ignoreFilter is False:
                    ocredtext = ocredtext.replace("\x0c", "")
                    parts = ocredtext.split(":")
                    if len(parts) > 1:
                        second_part = parts[1]
                        ocredtext=second_part

        finalImage = await self.write_text_to_image(
            finalImage, x_new, y_new-7, w_new, h_new, f"{ocredtext} (No-{validContourNo}) (TC-{icount}) (H-{h_new}) (P-{x_new},{y_new},{w_new},{h_new})"
        )

        if len(ocredtext)<4:
            return finalImage, False, (x_new, y_new, w_new, h_new)

        return finalImage, ocredtext , (x_new, y_new, w_new, h_new)
    
    async def write_text_to_image(self, image, x, y, w, h, text, font_color=(255, 0, 0)):

        # Draw the rectangle asynchronously
        if w>0 and h>0:
            # Generate a random color for the rectangle
            rect_color = tuple(np.random.randint(0, 255, 3).tolist())
            await asyncio.to_thread(
                cv2.rectangle, image, (x, y), (x + w, y + h), rect_color, 2
            )

        # Convert the image to RGB (OpenCV uses BGR by default)
        image_pil = await asyncio.to_thread(
            Image.fromarray, cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        )

        # Create a drawing context asynchronously
        draw = await asyncio.to_thread(ImageDraw.Draw, image_pil)

        # Loading the font (ensure the font file exists and the path is correct)
        try:
            font = await asyncio.to_thread(ImageFont.truetype, self.font_path, size=16)
        except IOError:
            raise FileNotFoundError(f"Font file not found at {self.font_path}")

        # Render the text asynchronously
        await asyncio.to_thread(draw.text, (x, y), text, font=font, fill=font_color)

        # Convert back to OpenCV format
        image = await asyncio.to_thread(cv2.cvtColor, np.array(image_pil), cv2.COLOR_RGB2BGR)

        return image
    
    async def direct_extract_text_from_image(self, image):
        # Perform OCR asynchronously on the provided image
        ocredtext = await asyncio.to_thread(pytesseract.image_to_string, image, lang = "eng+ben")
        
        # Strip unwanted characters and spaces
        ocredtext = ocredtext.strip()
        # ocredtext = ocredtext.replace("\n", "")
        # ocredtext = ocredtext.replace("\n", "<br>")

        return ocredtext


