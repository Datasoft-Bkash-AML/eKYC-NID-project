import os
import contextlib
import cv2
import numpy as np
import asyncio
from PIL import Image
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
# import torch
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="insightface.utils.transform")

# Suppress stdout and stderr
@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield

class ArcFaceMatcher:
    def __init__(self, threshold=80):
        try:
            with suppress_output():
                self.threshold = threshold
                self.app = FaceAnalysis(name='buffalo_s', providers=['CPUExecutionProvider'])
            # device_id = 0 if torch.cuda.is_available() else -1
            device_id = -1
            self.app.prepare(ctx_id=device_id)

        except Exception as e:
            print(f"Error on ArcFaceMatcher: {e}")

    async def _get_embedding(self, image_input):
        def _sync_embedding():
            try:
                if isinstance(image_input, str):
                    img = cv2.imread(image_input)
                elif isinstance(image_input, np.ndarray):
                    img = image_input
                elif isinstance(image_input, Image.Image):
                    img = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
                else:
                    raise ValueError("Unsupported image format")

                if img is None or img.shape[0] == 0 or img.shape[1] == 0:
                    print("Invalid image shape.")
                    return None, None, None

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=FutureWarning)
                    faces = self.app.get(img)

                if faces and hasattr(faces[0], 'embedding'):
                    face = faces[0]
                    embedding = face.embedding
                    x1, y1, x2, y2 = face.bbox.astype(int)
                    cropped_face = img[y1:y2, x1:x2]
                    face_rect = (x1, y1, x2, y2)
                    return embedding, cropped_face, face_rect
                else:
                    print("No face detected by InsightFace")
                    return None, None, None

            except Exception as e:
                print(f"Error processing face embedding: {e}")
                return None, None, None

        return await asyncio.to_thread(_sync_embedding)


    async def compare(self, profileImage, nidImage):
        (emb1, face1, rect1), (emb2, face2, rect2) = await asyncio.gather(
            self._get_embedding(profileImage),
            self._get_embedding(nidImage)
        )


        if (emb1 is None and emb2 is None) or (emb2 is None):
            return {
                "is_face_matched": False,
                "face_matched_score": 0.0,
                "face_matched_threshold": 0.0,
                "reason": "Face not detected",
                "profileFace": None,
                "nidFace": None,
                "profileFace_rect": None,
                "nidFace_rect": None
            }
        elif emb1 is None and emb2 is not None:
            return {
                "is_face_matched": False,
                "face_matched_score": 0.0,
                "face_matched_threshold": 0.0,
                "reason": "Face not detected",
                "profileFace": None,
                "nidFace": face2,
                "profileFace_rect": None,
                "nidFace_rect": rect2
            }

        similarity = cosine_similarity([emb1], [emb2])[0][0]
        actual_match_percent = similarity * 100
        match_percent = self.scale_to_100_range(actual_match_percent)
        is_match = True if match_percent >= self.threshold else False

        return {
            "profileFace": face1,
            "nidFace": face2,
            "profileFace_rect": rect1,
            "nidFace_rect": rect2,
            "is_face_matched": is_match,
            "face_matched_threshold": float(f"{actual_match_percent:.2f}"),
            "face_matched_score": float(f"{match_percent:.2f}")
        }

    def scale_to_100_range(self, value, max_original=70):
        if value < 1:
            value = 1
        if value > max_original:
            value = max_original
        
        scaled = (value / max_original) * 100
        return round(scaled, 2)
    
    # Synchronous validator
    def validate_face_image_sync(self, image_bytes: bytes) -> tuple[bool, str]:
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            return {
                "isMatched": False, 
                "message": f"Image could not be loaded."
            }

        faces = self.app.get(image)
        if len(faces) == 0:
            return {
                "isMatched": False, 
                "message": f"No face detected. Please try again with a well-lit, clear image of your face."
            }
        if len(faces) > 1:
            return {
                "isMatched": False, 
                "message": f"Image must contain exactly one face."
            }

        face = faces[0]

        # Check if face is frontal (pose angles close to 0)
        yaw, pitch, roll = face.pose
        if abs(yaw) > 20 or abs(pitch) > 20 or abs(roll) > 20:
            return False, f"Face is not frontal (yaw={yaw:.2f}, pitch={pitch:.2f}, roll={roll:.2f})"

        # Check if face size is large enough (e.g., >20% of image)
        bbox = face.bbox.astype(int)
        face_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        image_area = image.shape[0] * image.shape[1]
        face_ratio = face_area / image_area

        if face_ratio < 0.15:
            return {
                "isMatched": False, 
                "message": f"Face not detected or too small. Try uploading a clearer photo."
            }

        return  {
                "isMatched": True, 
                "message": f"Profile picture accepted. Looks good!"
            }