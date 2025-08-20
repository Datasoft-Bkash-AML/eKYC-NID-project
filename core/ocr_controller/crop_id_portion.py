from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

class CropProcessor:
    def __init__(self):
        self.model = YOLO('yolov8l.pt')
        self.backup_model = YOLO('yolov8l.pt')  # Backup model

    def multi_scale_preprocessing(self, image):
        """Generate multiple preprocessed versions of the image"""
        variants = {}

        # Original
        variants['original'] = image

        # 1. Contrast enhancement with multiple methods
        # CLAHE on LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l_clahe = clahe.apply(l)
        variants['clahe_lab'] = cv2.cvtColor(cv2.merge([l_clahe, a, b]), cv2.COLOR_LAB2BGR)

        # CLAHE on grayscale then convert back
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe_gray = clahe.apply(gray)
        variants['clahe_gray'] = cv2.cvtColor(clahe_gray, cv2.COLOR_GRAY2BGR)

        # 2. Histogram equalization
        yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
        variants['hist_eq'] = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

        # 3. Gamma correction (multiple values)
        for gamma in [0.7, 1.3]:
            gamma_corrected = np.power(image / 255.0, gamma) * 255.0
            variants[f'gamma_{gamma}'] = gamma_corrected.astype(np.uint8)

        # 4. Bilateral filtering (noise reduction while preserving edges)
        variants['bilateral'] = cv2.bilateralFilter(image, 9, 75, 75)

        # 5. Color space variations
        # Enhance saturation in HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[:,:,1] = np.clip(hsv[:,:,1] * 1.2, 0, 255)  # Increase saturation
        variants['enhanced_hsv'] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # 6. Edge-preserving filter
        variants['edge_preserve'] = cv2.edgePreservingFilter(image, flags=1, sigma_s=50, sigma_r=0.4)

        # 7. Unsharp masking for sharpening
        gaussian = cv2.GaussianBlur(image, (0, 0), 2.0)
        variants['unsharp'] = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)

        return variants

    def detect_with_multiple_models(self, image_variants):
        """Use multiple models and preprocessing variants"""
        all_detections = []

        models = [self.model, self.backup_model]
        confidence_thresholds = [0.15, 0.25, 0.35]  # Multiple thresholds

        for model_name, model in enumerate(models):
            for variant_name, image in image_variants.items():
                for conf_thresh in confidence_thresholds:
                    try:
                        results = model(image, conf=conf_thresh, verbose=False)

                        if len(results) > 0 and hasattr(results[0], 'boxes') and len(results[0].boxes) > 0:
                            for box in results[0].boxes:
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                                conf = float(box.conf[0].cpu().numpy())
                                cls = int(box.cls[0].cpu().numpy())

                                width = x2 - x1
                                height = y2 - y1
                                area = width * height
                                aspect_ratio = width / height if height > 0 else 0

                                # More comprehensive filtering
                                if self.is_valid_detection(width, height, area, aspect_ratio, cls, conf, image.shape):
                                    detection = {
                                        'bbox': (x1, y1, x2, y2),
                                        'confidence': conf,
                                        'class': cls,
                                        'aspect_ratio': aspect_ratio,
                                        'area': area,
                                        'variant': variant_name,
                                        'model': model_name,
                                        'conf_thresh': conf_thresh,
                                        'width': width,
                                        'height': height
                                    }
                                    all_detections.append(detection)
                    except Exception as e:
                        print(f"Error with {variant_name}, model {model_name}: {e}")
                        continue

        return all_detections

    def is_valid_detection(self, width, height, area, aspect_ratio, cls, conf, img_shape):
        """Enhanced validation for detections"""
        img_area = img_shape[0] * img_shape[1]
        relative_area = area / img_area

        # Size constraints
        if width < 30 or height < 20:  # Too small
            return False
        if relative_area > 0.8:  # Too large (probably whole image)
            return False
        if relative_area < 0.01:  # Too small relative to image
            return False

        # Aspect ratio constraints (ID cards are typically rectangular)
        if aspect_ratio < 0.8 or aspect_ratio > 2.5:  # Too narrow or too wide
            return False

        # Class-based filtering
        valid_classes = [
            73,  # book (most common for ID cards)
            0,   # person (sometimes detects person holding ID)
            67,  # cell phone (sometimes confused with ID)
            84,  # book (alternative book class)
            76,  # keyboard (sometimes confused)
        ]

        if cls not in valid_classes:
            return False

        # Confidence threshold (dynamic based on class)
        min_conf = 0.1 if cls == 73 else 0.15
        if conf < min_conf:
            return False

        return True

    def cluster_and_filter_detections(self, detections):
        """Use clustering to group similar detections and pick the best"""
        if not detections:
            return []

        # Extract features for clustering
        features = []
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            features.append([center_x, center_y, det['width'], det['height']])

        features = np.array(features)

        # Normalize features
        features_norm = features / np.max(features, axis=0)

        # Cluster detections
        if len(features) > 1:
            clustering = DBSCAN(eps=0.3, min_samples=1).fit(features_norm)
            labels = clustering.labels_
        else:
            labels = [0]

        # Group by clusters and pick best from each
        clusters = {}
        for i, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(detections[i])

        # Pick best detection from each cluster
        best_detections = []
        for cluster_detections in clusters.values():
            # Score each detection in the cluster
            scored_detections = []
            for det in cluster_detections:
                score = self.calculate_comprehensive_score(det)
                scored_detections.append((score, det))

            # Pick highest scoring detection from this cluster
            if scored_detections:
                best_score, best_det = max(scored_detections, key=lambda x: x[0])
                best_det['final_score'] = best_score
                best_detections.append(best_det)

        # Sort by final score
        best_detections.sort(key=lambda x: x['final_score'], reverse=True)

        return best_detections

    def calculate_comprehensive_score(self, detection):
        """More sophisticated scoring function"""
        score = 0.0

        # Base confidence score
        score += detection['confidence'] * 0.3

        # Class-based scoring
        class_scores = {
            73: 0.4,   # book - highest score
            0: 0.2,    # person
            67: 0.1,   # cell phone
            84: 0.35,  # book alternative
            76: 0.05   # keyboard
        }
        score += class_scores.get(detection['class'], 0.0)

        # Aspect ratio scoring (ID cards are typically 1.4-1.7 ratio)
        aspect = detection['aspect_ratio']
        if 1.4 <= aspect <= 1.7:
            score += 0.25
        elif 1.2 <= aspect <= 1.9:
            score += 0.15
        elif 1.0 <= aspect <= 2.2:
            score += 0.05

        # Size scoring (prefer reasonable sizes)
        area = detection['area']
        if 5000 <= area <= 50000:  # Good size range
            score += 0.2
        elif 2000 <= area <= 80000:  # Acceptable size range
            score += 0.1

        # Variant bonus (some preprocessing methods are more reliable)
        variant_bonuses = {
            'clahe_lab': 0.05,
            'clahe_gray': 0.03,
            'bilateral': 0.02,
            'enhanced_hsv': 0.02,
            'contour': 0.01 # Add a small bonus for contour detection
        }
        score += variant_bonuses.get(detection.get('variant', 'contour'), 0.0) # Use .get with default

        # Model bonus (primary model might be more reliable)
        if detection.get('model') == 0:  # Primary model (use .get in case 'model' is missing)
            score += 0.02

        return min(score, 1.0)  # Cap at 1.0

    def contour_based_detection(self, image):
        """Backup method using contour detection"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Multiple preprocessing approaches
        methods = [
            cv2.bilateralFilter(gray, 9, 75, 75),
            cv2.GaussianBlur(gray, (5, 5), 0),
            cv2.medianBlur(gray, 5)
        ]

        all_contours = []

        for processed in methods:
            # Edge detection with different parameters
            for thresh1, thresh2 in [(30, 100), (50, 150), (100, 200)]:
                edges = cv2.Canny(processed, thresh1, thresh2)
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                all_contours.extend(contours)

        # Analyze contours
        candidates = []
        for contour in all_contours:
            # Approximate contour
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            if len(approx) >= 4:  # At least 4 corners (rectangular-ish)
                x, y, w, h = cv2.boundingRect(contour)
                area = cv2.contourArea(contour)

                if area > 1000 and w > 50 and h > 30:  # Minimum size
                    aspect_ratio = w / h
                    if 0.8 < aspect_ratio < 2.5:  # Reasonable aspect ratio
                        candidates.append({
                            'bbox': (x, y, x+w, y+h),
                            'area': area,
                            'aspect_ratio': aspect_ratio,
                            'confidence': min(area / 10000, 0.8),  # Pseudo-confidence
                            'class': 73,  # Assume book class
                            'method': 'contour',
                            'variant': 'contour' # Add 'variant' key
                        })

        # Sort by area (larger is likely better)
        candidates.sort(key=lambda x: x['area'], reverse=True)

        return candidates[:3]  # Return top 3 candidates

    def process_image(self, image):
        print("  Generating image variants...")
        variants = self.multi_scale_preprocessing(image)

        # Step 2: Multi-model detection
        print("  Running multi-model detection...")
        all_detections = self.detect_with_multiple_models(variants)
        print(f"  Found {len(all_detections)} raw detections")
        for det in all_detections:
          print(f"    Class={det['class']}, Confidence={det['confidence']:.3f}, "
                f"Aspect={det['aspect_ratio']:.2f}")

        # Step 3: Cluster and filter
        print("  Clustering and filtering detections...")
        best_detections = self.cluster_and_filter_detections(all_detections)

        # Step 4: Backup contour detection if no good detections
        if not best_detections or (best_detections and best_detections[0]['final_score'] < 0.3):
            print("  Running backup contour detection...")
            contour_candidates = self.contour_based_detection(image)
            for candidate in contour_candidates:
                candidate['final_score'] = self.calculate_comprehensive_score(candidate)
            best_detections.extend(contour_candidates)

            # Re-sort all detections
            best_detections.sort(key=lambda x: x['final_score'], reverse=True)


        if best_detections:
            best = best_detections[0]
            print(f"  ✓ Best detection: Score={best['final_score']:.3f}, "
                  f"Class={best['class']}, Confidence={best.get('confidence', 0):.3f}, "
                  f"Aspect={best['aspect_ratio']:.2f}")

            # Crop and correct
            cropped, corrected = self.crop_and_correct(image, best)

            return {
                'success': True,
                'original': image,
                'cropped': cropped,
                'corrected': corrected,
                'detection': best,
                'all_detections': best_detections[:5]  # Keep top 5 for analysis
            }
        else:
            print("  ✗ No suitable detections found")
            return {'success': False, 'original': image}

    def crop_and_correct(self, image, detection):
        """Improved cropping and skew correction"""
        x1, y1, x2, y2 = detection['bbox']

        # Dynamic padding based on detection quality
        padding_factor = 0.05 if detection['final_score'] > 0.7 else 0.1

        width = x2 - x1
        height = y2 - y1

        pad_x = max(5, int(width * padding_factor))
        pad_y = max(5, int(height * padding_factor))

        # Crop with padding
        x1_crop = max(0, x1 - pad_x)
        y1_crop = max(0, y1 - pad_y)
        x2_crop = min(image.shape[1], x2 + pad_x)
        y2_crop = min(image.shape[0], y2 + pad_y)

        cropped = image[y1_crop:y2_crop, x1_crop:x2_crop]

        # Enhanced skew correction
        corrected = self.advanced_skew_correction(cropped)

        return cropped, corrected

    def advanced_skew_correction(self, image):
        """More robust skew correction"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Multiple edge detection approaches
        edge_methods = [
            cv2.Canny(cv2.bilateralFilter(gray, 9, 75, 75), 30, 100),
            cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 50, 150),
            cv2.Canny(gray, 100, 200)
        ]

        angles = []

        for edges in edge_methods:
            # Hough line detection with multiple parameters
            for threshold in [30, 50, 80]:
                lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=threshold)
                if lines is not None:
                    for line in lines[:10]:  # Top 10 lines
                        rho, theta = line[0]
                        angle = theta * 180 / np.pi
                        if angle > 90:
                            angle = angle - 180
                        if abs(angle) < 30:  # Only reasonable angles
                            angles.append(angle)

        if not angles:
            return image

        # Use median angle for robustness
        angle = np.median(angles)

        # Only correct if angle is significant
        if abs(angle) > 0.5:
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

            # Calculate new dimensions to avoid cropping
            cos_a = abs(rotation_matrix[0, 0])
            sin_a = abs(rotation_matrix[0, 1])
            new_w = int((h * sin_a) + (w * cos_a))
            new_h = int((h * cos_a) + (w * sin_a))

            # Adjust rotation matrix
            rotation_matrix[0, 2] += (new_w / 2) - center[0]
            rotation_matrix[1, 2] += (new_h / 2) - center[1]

            corrected = cv2.warpAffine(image, rotation_matrix, (new_w, new_h),
                                     flags=cv2.INTER_CUBIC,
                                     borderMode=cv2.BORDER_REPLICATE)
            return corrected

        return image
