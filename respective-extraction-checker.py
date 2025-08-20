# if name == self.nidFields[0]: # bng_name
#                 replacePattern = [(r'[^\u0980-\u09FF\s\.]', ''), (r'^\s*নাম\s*[:：]?\s*', '')]
#                 mainImage, ocrText = await self.extract_text(mainImage, detectImages, (x , y, w, h), name, unique_id,image_side, show_steps, "ben", False, replacePattern, r"--oem 1 --psm 7 -c preserve_interword_spaces=1")
#                 validContourCount += 1
            
#             elif name == self.nidFields[1]: # eng_name
#                 replacePattern = [(r'[^a-zA-Z\s\.]', ''), (r'.*?\bName\b\s*[:：]?\s*', ''), (r'\bmo\.\s*', 'MD. ')]
#                 mainImage, ocrText = await self.extract_text(mainImage, detectImages, (x , y, w, h), name, unique_id,image_side, show_steps, "eng", False, replacePattern, r"--oem 1 --psm 7 -c preserve_interword_spaces=1")
#                 validContourCount += 1

#             elif name == self.nidFields[2]: # father_name
#                 replacePattern = [(r'[^\u0980-\u09FF\s\.]', ''), (r'^\s*পিতা\s*[:：]?\s*', '')]
#                 mainImage, ocrText = await self.extract_text(mainImage, detectImages, (x , y, w, h), name, unique_id,image_side, show_steps, "ben", False, replacePattern, r"--oem 1 --psm 7 -c preserve_interword_spaces=1")
#                 validContourCount += 1
            
#             elif name == self.nidFields[3]: # mother_name
#                 replacePattern = [(r'[^\u0980-\u09FF\s\.]', ''), (r'^\s*মাতা\s*[:：]?\s*', ''), ]
#                 mainImage, ocrText = await self.extract_text(mainImage, detectImages, (x , y, w, h), name, unique_id,image_side, show_steps, "ben", False, replacePattern, r"--oem 1 --psm 7 -c preserve_interword_spaces=1")
#                 validContourCount += 1

#             elif name == self.nidFields[4]: # dob
#                 # dob_pattern = r'(?:[^\d]?)(\d{1,2}\s+[A-Za-z]{3}\s+(?:\d\s?){4})'
#                 dob_pattern = r'\b(0?[1-9]|[12][0-9]|3[01])[\s\-]+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[\s\-]+(\d{4})\b'
#                 replacePattern = [(r'[-\s]+', ' '), (r'\s+', ' ')]
#                 mainImage, ocrText = await self.extract_text(mainImage, detectImages, (x , y, w, h), name, unique_id,image_side, show_steps, "eng+digits", dob_pattern, replacePattern, r"--oem 1 --psm 7 -c preserve_interword_spaces=1")
#                 validContourCount += 1

#             elif name == self.nidFields[5]: # nid
#                 nid_pattern = r'\b(?:\d[-\s]?){9}(?:\d)\b|\b(?:\d[-\s]?){12}(?:\d)\b|\b(?:\d[-\s]?){16}(?:\d)\b'
#                 replacePattern = [(r'[-\s]+', ''),(r'[\s-]+', '') , (r'\s+', '')]
#                 mainImage, ocrText = await self.extract_text(mainImage, detectImages, (x , y, w, h), name, unique_id,image_side, show_steps, "eng+digits", nid_pattern, replacePattern, r"--oem 1 --psm 7")
#                 validContourCount += 1

#  async def extract_text(self, finalImage, detectImages, box, name, unique_id, image_side, show_steps, lang="ben+eng", regexPatten=False, replacePattern=[], config="--oem 1 --psm 7"):
#         icount = 0
#         ocredtext = ''
#         (x, y, w, h) = box

#         while icount < len(detectImages):
#             detectImage = detectImages[icount]
                    
#             # Crop the region of interest asynchronously
#             cropped = detectImage[y:y + h, x:x + w]
#             cropped = await self.downscale_image_async(cropped, 0.5)
#             # cropped_downscaled = await self.downscale_image_async(cropped, 0.7)
#             if show_steps:
#                 await asyncio.to_thread(cv2.imwrite, os.path.join(f"./history/{unique_id}/{image_side}", f"{name}_TC{icount}.jpg"), cropped)


#             # Perform OCR asynchronously
#             ocredtext = await asyncio.to_thread(
#                 pytesseract.image_to_string, cropped,  lang=lang , config=config
#                 # pytesseract.image_to_string, cropped,  lang=lang , config=r'--oem 1 --psm 7'
#             )
#             ocredtext = ocredtext.strip()
#             ocredtext = ocredtext.strip().replace("\n", "")

#             if regexPatten :
#                 match = re.search(regexPatten, ocredtext, re.IGNORECASE)
#                 if match:
#                     if match.lastgroup == 1:  # Check if group(1) exists
#                         ocredtext = match.group(1)
#                     else:
#                         ocredtext = match.group(0)
#                 else:
#                     ocredtext = ''

#             if len(replacePattern)>0:
#                 for pattern, replacement in replacePattern:
#                     ocredtext = re.sub(pattern, replacement, ocredtext, flags=re.IGNORECASE)
#                     # ocredtext = re.sub(r'^[^\u0980-\u09FFa-zA-Z0-9]+|[^\u0980-\u09FFa-zA-Z0-9]+$', '', ocredtext)

#             if len(ocredtext)>3:
#                 break
#             else:
#                 icount += 1

#             # Tried last time once by sharpening the image if no text detected
#             if (icount == len(detectImages)) and len(ocredtext)<4:
#                 sharped = await self.image_sharpener(cropped)
#                 sharped = await self.downscale_image_async(sharped, 0.5)
#                 if show_steps:
#                     await asyncio.to_thread(cv2.imwrite, os.path.join(f"./history/{unique_id}/{image_side}", f"{name}_TC{icount+1}.jpg"), sharped)

#                 # Perform OCR asynchronously
#                 # --oem 1: Uses LSTM OCR engine (good for most modern use cases)
#                 # --psm 7: Treats the image as a single text line (good for cropped fields like names)
                
#                 ocredtext = await asyncio.to_thread(
#                     pytesseract.image_to_string, sharped,  lang=lang , config=r'--oem 1 --psm 7'
#                 )
#                 ocredtext = ocredtext.strip()
#                 ocredtext = ocredtext.replace("\n", "")

#                 if regexPatten :
#                     match = re.search(regexPatten, ocredtext, re.IGNORECASE)
#                     if match:
#                         if match.lastgroup == 1:  # Check if group(1) exists
#                             ocredtext = match.group(1)
#                         else:
#                             ocredtext = match.group(0)
#                     else:
#                         ocredtext = ''

#         # ocredtext = re.sub(r'^\s*[A-Za-z]+\s*[:：]?\s*', '', ocredtext)

#         finalImage = await self.write_text_to_image(
#             finalImage, x, y, w, h, f"{ocredtext} ({name}) (TC-{icount}) (H-{h}) (P-{x},{y},{w},{h})"
#         )

#         if len(ocredtext)<4:
#             return finalImage, False

#         return finalImage, ocredtext

# When path to image is given, depending on the field type generate ocr text

# Input: /path/to/image.jpg
# Output: OCR text extracted from the image

nameEnPath = "/path/to/nameEn/image.jpg"
# dobPath = "/path/to/dob/image.jpg"