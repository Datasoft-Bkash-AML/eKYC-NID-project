
import_log = []
try:
    import cv2
except ImportError as e:
    import_log.append(f"cv2 import failed: {e}")
try:
    import imutils
except ImportError as e:
    import_log.append(f"imutils import failed: {e}")
try:
    import numpy as np
except ImportError as e:
    import_log.append(f"numpy import failed: {e}")
try:
    import pytesseract
    # use tesseract usr/bin for ubuntu
    pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
except ImportError as e:
    import_log.append(f"pytesseract import failed: {e}")
try:
    from PIL import ImageFont, ImageDraw, Image
except ImportError as e:
    import_log.append(f"PIL import failed: {e}")
import os
import re
import asyncio
import time
from collections import defaultdict
from utils.custom_helper import format_dob_for_ec
from .face_matcher_controller import ArcFaceMatcher  # Assuming you have this library

faceMatcher = ArcFaceMatcher(threshold=80)

class OCRProcessor:
    def __init__(self):
        # current_folder = os.getcwd()
        current_folder = os.path.dirname(os.path.abspath(__file__))  # Get the current script's directory
        parent_folder = os.path.dirname(current_folder)
        parent_folder = os.path.dirname(parent_folder)
        target_folder = os.path.join(parent_folder, "assets")
        self.font_path = os.path.join(target_folder, 'ben.ttf')
        # Define the meaningful names for lines
        # self.nidFields = ["bng_name", "eng_name", "father_name", "mother_name", "dob", "nid_num"]
        self.nidFields = ["name", "nameEn", "father", "mother", "dateOfBirth", "nationalId"]

    async def estimate_image_quality(self, image_path: str, unique_id: str, high_accuracy: bool, is_nid_dob_extractable: bool) -> dict:
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

        requiredTextRegionScore = 1000 if high_accuracy else 500
        # --- Scoring System ---
        sharpness_score = 1 if sharpness > 1000 else 0.3
        contrast_score = 1 if contrast > 50 else 0.5
        brightness_score = 1 if 60 < brightness < 150 else 0.3
        text_region_score = 1 if text_like_regions > requiredTextRegionScore else 0
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

        acceptableScore = 100

        # if score >= 90:
        #     message = "Image is excellent for OCR"
        if score >= 70 and text_region_score == 1:
            message = "Looks good! This image is suitable for OCR."
        elif text_region_score == 0:
            acceptableScore = 0
            message = "Text quality is too low for OCR. Please provide a clearer image."
        else:
            acceptableScore = 50
            message = "Image quality may affect OCR accuracy."

        if is_nid_dob_extractable:
            result = await self.processFrontSide(image_path, unique_id, image_path, True)
            if(result and result['nid_data'] and (len(result['nid_data'][self.nidFields[4]])<10 or len(result['nid_data'][self.nidFields[5]])<10)):
                message="Unable to extract text from the image. Please upload a clearer and higher-quality image."
                acceptableScore=-100
            
        return {
            "sharpness": round(sharpness, 2),
            "contrast": round(contrast, 2),
            "brightness": round(brightness, 2),
            "text_like_regions": text_like_regions,
            "glare_ratio": round(glare_ratio, 4),
            "skew_angle_deg": round(skew_angle, 2),
            "score": round(score, 2),
            "message": message,
            "acceptableScore": acceptableScore
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

        resizedImage = await asyncio.to_thread(imutils.resize, nidImage, height=1000)
        
        faceMatchingResult = await faceMatcher.compare(profileImage,resizedImage)

        resizedCopyImage = resizedImage.copy()

        gray = await asyncio.to_thread(cv2.cvtColor, resizedCopyImage, cv2.COLOR_BGR2GRAY)
        
        # Face detection
        faceRectangles = (0, 0, 0, 0)
        
        if faceMatchingResult.get("nidFace_rect") is not None:
            x, y, w, h = faceMatchingResult.get("nidFace_rect")

            # Add extra space to include more context around the face
            extra_space = int(w * 0.15)

            x_new = (x - extra_space)
            w_new = (w + extra_space)
            y_new = (y - extra_space)
            h_new = (h + extra_space)

            faceRectangles = (x_new, y_new, w_new, h_new)
            face = resizedImage[y_new:h_new, x_new:w_new]
            face_height, _ = face.shape[:2]

            if show_steps:
                if faceMatchingResult.get("profileFace") is not None and faceMatchingResult.get("profileFace").any():
                    combined_img = await self.process_images_into_one(face, [faceMatchingResult.get("profileFace"),faceMatchingResult.get("nidFace")])  
                else:
                    combined_img = await self.process_images_into_one(face, [faceMatchingResult.get("nidFace")])  

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

            if area < 1500:  # Apply your threshold condition
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
        detectImages = [gray, resizedCopyImage, grayNew, enhanced]
        
        try:
            gray_finalImage,ocrResults = await self.start_ocr_frontSide(resizedImage, detectImages, contours, faceRectangles, show_steps, unique_id)
        except Exception as e:
            return {
                    "success": False,
                    "message": "Request not successful",
                    "error": "OCR failed to read the NID card. Please upload a clearer, high-quality image."
                }

        if show_steps:
            # Draw rectangle around the largest face asynchronously
            if faceRectangles != (0, 0, 0, 0):
                (x_f, y_f, w_f, h_f) = faceRectangles
                await asyncio.to_thread(
                    cv2.rectangle, gray_finalImage, (x_f, y_f), (w_f, h_f), (0, 255, 0), 2
                )
            # detectImages = detectImages + [close, gray_finalImage, face]
            detectImages = detectImages + [blackhatNew, gradXNew, threshNew, close, gray_finalImage]
            # detectImages = detectImages[:3] + [blackhatNew, gradXNew, threshNew, close, gray_finalImage]
            # detectImages = detectImages[:3] + [gray_finalImage]
            
            combined_img = await self.process_images_into_one(resizedImage, detectImages)  
            output_path = os.path.join(f"./history/{unique_id}/FrontSide", f"FrontSide.Processed.jpg")
            await asyncio.to_thread(cv2.imwrite, output_path, combined_img)
        
        if "face_data" not in ocrResults:
                ocrResults["face_data"] = {}

        for field in self.nidFields:
            if field not in ocrResults['nid_data'] or ocrResults['nid_data'][field] is None:
                ocrResults['nid_data'][field] = ""
                
        ocrResults['face_data']['is_face_matched'] = faceMatchingResult['is_face_matched']
        ocrResults['face_data']['face_matched_score'] = faceMatchingResult['face_matched_score']
        ocrResults['face_data']['face_matched_threshold'] = faceMatchingResult['face_matched_threshold']
        
        return ocrResults

    async def start_ocr_frontSide(self, mainImage, detectImages, contours, faceRectangles, show_steps, unique_id):

            image_side = "FrontSide"
            extra_space = 12
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
                
                if 1.5 < aspect_ratio and 1500 < area < 35000 and (x2-(x2/4)) < x and (y1-h)<y:
                    # print(f"Countour {index} values - x: {x}, y: {y}, w: {w}, h: {h}, area: {area} -> Accepted")
                    
                    # if w > max_width:
                    #     max_width = w

                    avgH += h
                    validContour += 1
                    valid_contours.append(contour)
                    # Set first contour if not already set
                    if first_contour is None:
                        first_contour = (x, y+abs(int(h/3)), w, h)
                        # first_contour = (x, y, w, h)

                    # Always update the last contour
                    last_contour = (x, y-abs(int(h/3)), w, h)
                
            line_positions = {}
            
            # Calculate height/distance between the first and last contours and create possible text zone
            if first_contour and last_contour:
                first_y = first_contour[1]
                last_y = last_contour[1]
                last_h = last_contour[3]

                # Calculate the distance
                distance = abs((last_y + last_h) - (first_y))

                step_size = distance // 5
                
                x_line = int(mainImage.shape[1]/1.6)
                for i in range(0, 6):
                    y_position = first_y + i * step_size
                    if show_steps:
                        cv2.line(mainImage, (x_line, y_position), (mainImage.shape[1], y_position), (0, 0, 255), 1)
                        cv2.putText(
                            mainImage,f"{self.nidFields[i]}-{y_position}",(mainImage.shape[1]-150, y_position - 5),cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,  # Font scale
                            (0, 0, 255),  # Font color (green)
                            1,  # Thickness
                            cv2.LINE_AA  # Line type
                        )
                    line_positions[self.nidFields[i]] = y_position


            avgH = avgH / validContour if validContour > 0 else 0

            mainImage = await self.write_text_to_image(
                mainImage, 10, 30, 0, 0, f"Details (avgH-{avgH}) (validContour-{validContour})"
            )

            validContourCount = 0
            validContoursGroup = []

            for i, contour in enumerate(valid_contours):
                x, y, w, h = cv2.boundingRect(contour)
                
                # possible_width = abs(int((img_width)*0.80))
                # if possible_width > x and possible_width > w:
                #     w = possible_width-x

                # if h > (avgH * 1.6):
                #     h = h // 2
                #     validContoursGroup.append((x , y, w, h))
                #     validContourCount += 1

                #     validContoursGroup.append((x , y+h+extra_space, w, h))
                #     validContourCount += 1
                # else:
                #     validContoursGroup.append((x , y, w, h))
                #     validContourCount += 1
                if h > (avgH * 1.6):
                    h = h // 2
                    validContoursGroup.append((x , y-extra_space, w, h+extra_space))
                    validContourCount += 1

                    validContoursGroup.append((x , y+h-extra_space, w, h+extra_space))
                    validContourCount += 1
                else:
                    validContoursGroup.append((x , y-extra_space, w, h+extra_space))
                    validContourCount += 1

            # Final grouped result
            line_to_contours = defaultdict(list)

            for _, (x, y, w, h) in enumerate(validContoursGroup):
                
                x = x - extra_space 
                y = y - extra_space 
                w = w + (3*extra_space) 
                h = h + (2*extra_space)

                bottom = y + h

                for name, line_y in line_positions.items():
                    if y <= line_y <= bottom:
                        line_to_contours[name].append((x, y, w, h))

            # merging contours
            for name in self.nidFields:
                boxes = line_to_contours.get(name, [])

                if not boxes:
                    continue

                x_min = min(x for x, y, w, h in boxes)
                y_min = min(y for x, y, w, h in boxes)
                x_max = max(x + w for x, y, w, h in boxes)
                y_max = max(y + h for x, y, w, h in boxes)

                merged_x = x_min
                merged_y = y_min
                merged_w = x_max - x_min
                merged_h = y_max - y_min

                # Replace the list of boxes with a single merged box
                line_to_contours[name] = (merged_x, merged_y, merged_w, merged_h)

            mainImage, ocrResults = await self.continue_ocr_frontside(mainImage, line_to_contours,detectImages,unique_id, image_side, show_steps)
                    
            return mainImage , ocrResults


    async def continue_ocr_frontside(self, mainImage, line_to_contours, detectImages, unique_id, image_side, show_steps):
        ocrResults = {}
        validContourCount, min_x = 0 , 0
        
        # Calculate average x for first 4 items
        selected_fields = {
            name: (x, y, w, h)
            for name, (x, y, w, h) in line_to_contours.items()
            if name in [self.nidFields[0],self.nidFields[1],self.nidFields[2],self.nidFields[3]]
        }

        if selected_fields:
            min_x = min(x for x, y, w, h in selected_fields.values())
            
        # First, find the maximum width among the target fields
        max_w = max(
            w for name, (x, y, w, h) in line_to_contours.items()
            if name in [self.nidFields[0], self.nidFields[1], self.nidFields[2], self.nidFields[3]]
        )

        _, img_width = mainImage.shape[:2]

        for name, (x, y, w, h) in line_to_contours.items():
            # if min_x <= x and name in [self.nidFields[0],self.nidFields[1],self.nidFields[2],self.nidFields[3]]:
            if name in [self.nidFields[0],self.nidFields[1],self.nidFields[2],self.nidFields[3]]:
                x = min_x
                w = max_w

            if name in [self.nidFields[4],self.nidFields[5]]:
                # temp_width = abs(int((w)*0.30)) + w
                # if (temp_width + x) <= img_width:
                #     w = temp_width
                temp_width = abs(int((img_width)*0.90))
                if (w + x) <= temp_width:
                    w = temp_width - x
                
            ocrText = None

            if name == self.nidFields[0]: # bng_name
                replacePattern = [(r'[^\u0980-\u09FF\s\.]', ''), (r'^\s*নাম\s*[:：]?\s*', '')]
                mainImage, ocrText = await self.extract_text(mainImage, detectImages, (x , y, w, h), name, unique_id,image_side, show_steps, "ben", False, replacePattern, r"--oem 1 --psm 7 -c preserve_interword_spaces=1")
                validContourCount += 1
            
            elif name == self.nidFields[1]: # eng_name
                replacePattern = [(r'[^a-zA-Z\s\.]', ''), (r'.*?\bName\b\s*[:：]?\s*', ''), (r'\bmo\.\s*', 'MD. ')]
                mainImage, ocrText = await self.extract_text(mainImage, detectImages, (x , y, w, h), name, unique_id,image_side, show_steps, "eng", False, replacePattern, r"--oem 1 --psm 7 -c preserve_interword_spaces=1")
                validContourCount += 1

            elif name == self.nidFields[2]: # father_name
                replacePattern = [(r'[^\u0980-\u09FF\s\.]', ''), (r'^\s*পিতা\s*[:：]?\s*', '')]
                mainImage, ocrText = await self.extract_text(mainImage, detectImages, (x , y, w, h), name, unique_id,image_side, show_steps, "ben", False, replacePattern, r"--oem 1 --psm 7 -c preserve_interword_spaces=1")
                validContourCount += 1
            
            elif name == self.nidFields[3]: # mother_name
                replacePattern = [(r'[^\u0980-\u09FF\s\.]', ''), (r'^\s*মাতা\s*[:：]?\s*', ''), ]
                mainImage, ocrText = await self.extract_text(mainImage, detectImages, (x , y, w, h), name, unique_id,image_side, show_steps, "ben", False, replacePattern, r"--oem 1 --psm 7 -c preserve_interword_spaces=1")
                validContourCount += 1

            elif name == self.nidFields[4]: # dob
                # dob_pattern = r'(?:[^\d]?)(\d{1,2}\s+[A-Za-z]{3}\s+(?:\d\s?){4})'
                dob_pattern = r'\b(0?[1-9]|[12][0-9]|3[01])[\s\-]+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[\s\-]+(\d{4})\b'
                replacePattern = [(r'[-\s]+', ' '), (r'\s+', ' ')]
                mainImage, ocrText = await self.repeatetive_process_text_for_nid_no(mainImage, detectImages, (x , y, w, h), name, unique_id,image_side, show_steps, "eng+digits", dob_pattern, replacePattern, r"--oem 1 --psm 7 -c preserve_interword_spaces=1")
                validContourCount += 1

            elif name == self.nidFields[5]: # nid
                nid_pattern = r'\b(?:\d[-\s]?){9}(?:\d)\b|\b(?:\d[-\s]?){12}(?:\d)\b|\b(?:\d[-\s]?){16}(?:\d)\b'
                replacePattern = [(r'[-\s]+', ''),(r'[\s-]+', '') , (r'\s+', '')]
                mainImage, ocrText = await self.repeatetive_process_text_for_nid_no(mainImage, detectImages, (x , y, w, h), name, unique_id,image_side, show_steps, "eng+digits", nid_pattern, replacePattern, r"--oem 1 --psm 7 -c preserve_interword_spaces=1")
                validContourCount += 1

            if ocrText:
                # # Remove special characters but allow spaces within the text
                # ocrText = re.sub(r'[^a-zA-Z0-9\u0980-\u09FF ]', '', ocrText)
                # Remove spaces from the beginning and end
                ocrText = ocrText.strip()

                if name == self.nidFields[4]: # dob
                    ocrText = format_dob_for_ec(ocrText)
            else:
                ocrText = ''

            if "nid_data" not in ocrResults:
                ocrResults["nid_data"] = {}

            ocrResults["nid_data"][name] = ocrText

        return mainImage, ocrResults


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
    
    async def extract_text(self, finalImage, detectImages, box, name, unique_id, image_side, show_steps, lang="ben+eng", regexPatten=False, replacePattern=[], config="--oem 1 --psm 7"):
        icount = 0
        ocredtext = ''
        (x, y, w, h) = box

        while icount < len(detectImages):
            detectImage = detectImages[icount]
                    
            # Crop the region of interest asynchronously
            cropped = detectImage[y:y + h, x:x + w]
            cropped = await self.downscale_image_async(cropped, 0.5)
            # cropped_downscaled = await self.downscale_image_async(cropped, 0.7)
            if show_steps:
                await asyncio.to_thread(cv2.imwrite, os.path.join(f"./history/{unique_id}/{image_side}", f"{name}_TC{icount}.jpg"), cropped)


            # Perform OCR asynchronously
            ocredtext = await asyncio.to_thread(
                pytesseract.image_to_string, cropped,  lang=lang , config=config
                # pytesseract.image_to_string, cropped,  lang=lang , config=r'--oem 1 --psm 7'
            )
            ocredtext = ocredtext.strip()
            ocredtext = ocredtext.strip().replace("\n", "")

            if regexPatten :
                match = re.search(regexPatten, ocredtext, re.IGNORECASE)
                if match:
                    if match.lastgroup == 1:  # Check if group(1) exists
                        ocredtext = match.group(1)
                    else:
                        ocredtext = match.group(0)
                else:
                    ocredtext = ''

            if len(replacePattern)>0:
                for pattern, replacement in replacePattern:
                    ocredtext = re.sub(pattern, replacement, ocredtext, flags=re.IGNORECASE)
                    # ocredtext = re.sub(r'^[^\u0980-\u09FFa-zA-Z0-9]+|[^\u0980-\u09FFa-zA-Z0-9]+$', '', ocredtext)
                    

            if len(ocredtext)>3:
                break
            else:
                icount += 1

            # Tried last time once by sharpening the image if no text detected
            if (icount == len(detectImages)) and len(ocredtext)<4:
                sharped = await self.image_sharpener(cropped)
                sharped = await self.downscale_image_async(sharped, 0.5)
                if show_steps:
                    await asyncio.to_thread(cv2.imwrite, os.path.join(f"./history/{unique_id}/{image_side}", f"{name}_TC{icount+1}.jpg"), sharped)

                # Perform OCR asynchronously
                # --oem 1: Uses LSTM OCR engine (good for most modern use cases)
                # --psm 7: Treats the image as a single text line (good for cropped fields like names)
                
                ocredtext = await asyncio.to_thread(
                    pytesseract.image_to_string, sharped,  lang=lang , config=r'--oem 1 --psm 7'
                )
                ocredtext = ocredtext.strip()
                ocredtext = ocredtext.replace("\n", "")

                if regexPatten :
                    match = re.search(regexPatten, ocredtext, re.IGNORECASE)
                    if match:
                        if match.lastgroup == 1:  # Check if group(1) exists
                            ocredtext = match.group(1)
                        else:
                            ocredtext = match.group(0)
                    else:
                        ocredtext = ''

        # ocredtext = re.sub(r'^\s*[A-Za-z]+\s*[:：]?\s*', '', ocredtext)

        finalImage = await self.write_text_to_image(
            finalImage, x, y, w, h, f"{ocredtext} ({name}) (TC-{icount}) (H-{h}) (P-{x},{y},{w},{h})"
        )

        if len(ocredtext)<4:
            return finalImage, False

        return finalImage, ocredtext
    

    async def downscale_image_async(self, img, scale_factor=0.5):
        def resize_image():
            height, width = img.shape[:2]
            new_size = (int(width * scale_factor), int(height * scale_factor))
            return cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)  # Best for downscaling

        return await asyncio.to_thread(resize_image)

    async def process_text(self, finalImage, detectImages, x, y, w, h, validContourNo,unique_id,image_side,show_steps, lang="ben+eng", ignoreFilter=False):
        
        icount = 0
        ocredtext = ''

        while icount < len(detectImages):
            detectImage = detectImages[icount]
                    
            x_new = max(x, 0)
            w_new = w
            y_new = max(y, 0)
            h_new = h
            
            # Crop the region of interest asynchronously
            cropped = detectImage[y_new:y_new + h_new, x_new:x_new + w_new]
            
            if show_steps:
                await asyncio.to_thread(cv2.imwrite, os.path.join(f"./history/{unique_id}/{image_side}", f"No{validContourNo}_TC{icount}.jpg"), cropped)

            # Perform OCR asynchronously
            ocredtext = await asyncio.to_thread(
                pytesseract.image_to_string, cropped, lang=lang , config=r'--oem 1 --psm 7'
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
                    pytesseract.image_to_string, sharped,  lang=lang , config=r'--oem 1 --psm 7'
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
            return finalImage, False

        return finalImage, ocredtext
    
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
        image = cv2.imread(str(image))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
        # Thresholding
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        
        # Denoising
        denoised = cv2.fastNlMeansDenoising(thresh, h=10)
        
        ocredtext = await asyncio.to_thread(pytesseract.image_to_string, denoised, lang = "eng+ben", config=r'--oem 1 --psm 6')
        
        # Strip unwanted characters and spaces
        ocredtext = ocredtext.strip()
        # ocredtext = ocredtext.replace("\n", "")
        # ocredtext = ocredtext.replace("\n", "<br>")

        return ocredtext


