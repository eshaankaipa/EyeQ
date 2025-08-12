import cv2
import numpy as np
import os
import time
from datetime import datetime
import sys
import re
import difflib
import ssl
import requests
import base64
import os
from typing import List, Tuple, Optional
import threading
import queue
import json

# Try to import OCR libraries
try:
    import easyocr
    OCR_AVAILABLE = True
    print("✓ EasyOCR available for text detection")
except ImportError:
    try:
        import pytesseract
        OCR_AVAILABLE = True
        print("✓ Tesseract available for text detection")
    except ImportError:
        OCR_AVAILABLE = False
        print("⚠ No OCR library available. Install easyocr or pytesseract for text detection.")

class TextDetectionCamera:
    def __init__(self):
        self.camera = None
        self.captured_images = []
        self.output_dir = "captured_images"
        self.running = True
        self.ocr_reader = None
        self.detection_enabled = True
        self.use_fallback = False  # Use fallback detection method
        # Prefer accurate OCR mode by default when OCR is available
        self.fast_mode = False
        self.accurate_mode = True
        # Store last detected regions as list of dicts: {x,y,w,h,score,label}
        self.last_text_regions: List[dict] = []
        # Async detection infrastructure
        self._det_queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=1)
        self._det_thread: Optional[threading.Thread] = None
        self._det_stop = threading.Event()
        self._det_last_time: float = 0.0
        self._det_interval_sec: float = 0.15  # run OCR roughly ~6-7 Hz
        # Google Vision client (optional)
        self.vision_client = None
        # Detection granularity: 'paragraph' | 'line' | 'word'
        # Default to word-level boxes to avoid one giant box for the entire problem
        self.granularity = 'word'
        # Center region fraction (width/height of valid detection area). Smaller than 0.75 now.
        self.center_fraction: float = 0.62
        # Autosolve (hands-free) state
        self._hold_signature: Optional[str] = None
        self._hold_start_time: float = 0.0
        self._autosolve_in_progress: bool = False
        # Autosolve parameters and debounce to ignore refocus/noise
        self.autosolve_hold_seconds: float = 3.0
        self.autosolve_change_debounce_seconds: float = 0.6
        self._pending_signature: Optional[str] = None
        self._pending_since: float = 0.0
        # Store last full recognized text (from Vision)
        self.last_full_text: str = ""
        # Last LLM solution to overlay on camera
        self.last_llm_solution_text: Optional[str] = None
        self.last_llm_solution_time: float = 0.0
        # Cooldown period after solution (5 seconds)
        self.solution_cooldown_seconds: float = 5.0
        # Mathpix (specialized math OCR) optional
        self.mathpix_app_id = os.getenv('MATHPIX_APP_ID')
        self.mathpix_app_key = os.getenv('MATHPIX_APP_KEY')
        self.mathpix_enabled = bool(self.mathpix_app_id and self.mathpix_app_key)
        if self.mathpix_enabled:
            print("✓ Mathpix OCR enabled for math symbols")
        
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # Initialize OCR if available
        if OCR_AVAILABLE:
            self.initialize_ocr()
        # Initialize Google Vision if credentials provided
        self.initialize_google_vision()
        
        # Enable OpenCV optimizations for better FPS
        try:
            cv2.setUseOptimized(True)
        except Exception:
            pass
    
    def initialize_ocr(self):
        """Initialize OCR reader for text detection with SSL fix"""
        try:
            if 'easyocr' in sys.modules:
                print("Initializing EasyOCR...")
                
                # Fix SSL certificate issue
                try:
                    # Create unverified SSL context
                    ssl._create_default_https_context = ssl._create_unverified_context
                    
                    self.ocr_reader = easyocr.Reader(['en'], gpu=False, download_enabled=True)
                    print("✓ EasyOCR initialized successfully!")
                    return
                except Exception as e:
                    print(f"EasyOCR failed to initialize: {e}")
                    print("Falling back to basic text detection...")
                    self.use_fallback = True
                    self.ocr_reader = 'fallback'
                    
            elif 'pytesseract' in sys.modules:
                print("✓ Tesseract OCR available")
                self.ocr_reader = 'tesseract'
                
        except Exception as e:
            print(f"Error initializing OCR: {e}")
            print("Using fallback text detection method...")
            self.use_fallback = True
            self.ocr_reader = 'fallback'
    
    def initialize_google_vision(self):
        """Initialize Google Cloud Vision client if credentials are available.
        Requires env var GOOGLE_APPLICATION_CREDENTIALS pointing to a JSON key
        or running in an environment with application default credentials.
        """
        try:
            from google.cloud import vision
            # If env var not set, try to auto-discover a service account JSON in CWD
            creds_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS', '').strip()
            if not creds_path or not os.path.isfile(creds_path):
                try:
                    for fname in os.listdir('.'):
                        if fname.lower().endswith('.json') and os.path.isfile(fname):
                            try:
                                with open(fname, 'r') as f:
                                    data = json.load(f)
                                if isinstance(data, dict) and data.get('type') == 'service_account':
                                    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.path.abspath(fname)
                                    print(f"✓ Using Google Vision credentials: {fname}")
                                    break
                            except Exception:
                                continue
                except Exception:
                    pass

            # Attempt to create a client; will raise if creds not available/invalid
            self.vision_client = vision.ImageAnnotatorClient()
            print("✓ Google Cloud Vision initialized")
        except Exception as e:
            self.vision_client = None
            # Print detailed reason to help diagnose setup issues
            creds_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS', '(not set)')
            print(f"Google Vision init failed: {e}\n  GOOGLE_APPLICATION_CREDENTIALS={creds_path}\nUsing EasyOCR fallback.")

    def google_vision_ocr(self, frame_bgr):
        """Use Google Cloud Vision to detect text and return boxes at selected granularity
        ('paragraph', 'line', or 'word'). Returns (annotated_frame, regions_list)
        where each region is {x,y,w,h,score,label}.
        Uses geometry-preserving pre-processing and reconstructs lines using symbol-level breaks
        to better capture math symbols like '+', '=', and '≥'.
        """
        try:
            if self.vision_client is None:
                return frame_bgr, []
            from google.cloud import vision

            # Geometry-preserving pre-processing for OCR (contrast/sharpness)
            proc = self.preprocess_for_vision(frame_bgr)

            # Encode as PNG (lossless) to avoid introducing JPEG artifacts
            success, buf = cv2.imencode('.png', proc)
            if not success:
                return frame_bgr, []

            # Build explicit request to set latest model and enable confidence scores
            image = vision.Image(content=buf.tobytes())
            features = [vision.Feature(type_=vision.Feature.Type.DOCUMENT_TEXT_DETECTION, model="builtin/latest")]
            image_context = vision.ImageContext(
                language_hints=['en'],
                text_detection_params=vision.TextDetectionParams(enable_text_detection_confidence_score=True)
            )
            request = vision.AnnotateImageRequest(image=image, features=features, image_context=image_context)
            batch_response = self.vision_client.batch_annotate_images(requests=[request])
            response = batch_response.responses[0]
            if response.error.message:
                print(f"Google Vision error: {response.error.message}")
                return frame_bgr, []

            height, width = frame_bgr.shape[:2]
            if not (response.full_text_annotation and response.full_text_annotation.pages):
                return frame_bgr, []

            regions = []
            # Helpers to get bbox from Vision vertices
            def bbox_to_xywh(vertices):
                xs = [v.x for v in vertices]
                ys = [v.y for v in vertices]
                x0, y0 = max(0, min(xs)), max(0, min(ys))
                x1, y1 = min(width, max(xs)), min(height, max(ys))
                return int(x0), int(y0), int(max(0, x1 - x0)), int(max(0, y1 - y0))

            # Build full text and line boxes using detected breaks
            full_text_parts = []
            for page in response.full_text_annotation.pages:
                for block in page.blocks:
                    for para in block.paragraphs:
                        # Paragraph-level region
                        if self.granularity == 'paragraph':
                            xs = [v.x for v in para.bounding_box.vertices]
                            ys = [v.y for v in para.bounding_box.vertices]
                            x0, y0 = max(0, min(xs)), max(0, min(ys))
                            x1, y1 = min(width, max(xs)), min(height, max(ys))
                            w, h = int(max(0, x1 - x0)), int(max(0, y1 - y0))
                            # Build paragraph text
                            words_txt = []
                            for word in para.words:
                                t = ''.join(sym.text for sym in word.symbols)
                                if t:
                                    words_txt.append(t)
                            text_para = self.normalize_math_text(' '.join(words_txt))
                            if w > 0 and h > 0 and text_para:
                                pad_x = int(max(4, w * 0.06))
                                pad_y = int(max(4, h * 0.06))
                                x = max(0, x0 - pad_x)
                                y = max(0, y0 - pad_y)
                                w2 = min(width - x, w + 2 * pad_x)
                                h2 = min(height - y, h + 2 * pad_y)
                                if self.is_valid_text_region(w2, h2) and self.is_in_center_region(x, y, w2, h2, width, height):
                                    regions.append({"x": x, "y": y, "w": w2, "h": h2, "score": 0.94, "label": text_para})

                        # Word-level regions
                        if self.granularity == 'word':
                            word_regions: List[dict] = []
                            for word in para.words:
                                t = ''.join(sym.text for sym in word.symbols)
                                if not t:
                                    continue
                                t_norm = self.normalize_math_text(t)
                                xw, yw, ww, hw = bbox_to_xywh(word.bounding_box.vertices)
                                if ww <= 0 or hw <= 0 or not t_norm:
                                    continue
                                pad_x = int(max(1, ww * 0.05))
                                pad_y = int(max(1, hw * 0.06))
                                x = max(0, xw - pad_x)
                                y = max(0, yw - pad_y)
                                w2 = min(width - x, ww + 2 * pad_x)
                                h2 = min(height - y, hw + 2 * pad_y)
                                if self.is_valid_word_region(w2, h2) and self.is_in_center_region(x, y, w2, h2, width, height):
                                    word_regions.append({"x": x, "y": y, "w": w2, "h": h2, "score": 0.9, "label": t_norm})
                            # Merge likely subscript tokens into their base variable tokens
                            word_regions = self.merge_subscript_word_boxes(word_regions)
                            regions.extend(word_regions)

                        # Line reconstruction (for last_full_text, and line-level regions when selected)
                        current_line_text = []
                        current_line_vertices = []
                        def flush_line():
                            if not current_line_text or not current_line_vertices:
                                return
                            txt = ''.join(current_line_text).strip()
                            norm_txt = self.normalize_math_text(txt)
                            if norm_txt:
                                xs = [v.x for v in current_line_vertices]
                                ys = [v.y for v in current_line_vertices]
                                x0, y0 = max(0, min(xs)), max(0, min(ys))
                                x1, y1 = min(width, max(xs)), min(height, max(ys))
                                w, h = int(max(0, x1 - x0)), int(max(0, y1 - y0))
                                pad_x = int(max(4, w * 0.07))
                                pad_y = int(max(4, h * 0.09))
                                x = max(0, int(x0 - pad_x))
                                y = max(0, int(y0 - pad_y))
                                w2 = min(width - x, int(w + 2 * pad_x))
                                h2 = min(height - y, int(h + 2 * pad_y))
                                if self.is_in_center_region(x, y, w2, h2, width, height):
                                    full_text_parts.append(norm_txt)
                            if self.granularity == 'line' and norm_txt:
                                xs = [v.x for v in current_line_vertices]
                                ys = [v.y for v in current_line_vertices]
                                x0, y0 = max(0, min(xs)), max(0, min(ys))
                                x1, y1 = min(width, max(xs)), min(height, max(ys))
                                w, h = int(max(0, x1 - x0)), int(max(0, y1 - y0))
                                pad_x = int(max(4, w * 0.07))
                                pad_y = int(max(4, h * 0.09))
                                x = max(0, int(x0 - pad_x))
                                y = max(0, int(y0 - pad_y))
                                w2 = min(width - x, int(w + 2 * pad_x))
                                h2 = min(height - y, int(h + 2 * pad_y))
                                if self.is_valid_text_region(w2, h2) and self.is_in_center_region(x, y, w2, h2, width, height):
                                    regions.append({"x": x, "y": y, "w": w2, "h": h2, "score": 0.95, "label": norm_txt})
                            current_line_text.clear()
                            current_line_vertices.clear()

                        for word in para.words:
                            # Append each symbol and consider its break type for line reconstruction
                            for sym in word.symbols:
                                t = sym.text or ''
                                current_line_text.append(t)
                                current_line_vertices.extend(sym.bounding_box.vertices)
                                br = getattr(sym, 'property', None)
                                brk = getattr(br, 'detected_break', None)
                                brk_type = getattr(brk, 'type_', None)
                                if brk_type in (vision.TextAnnotation.DetectedBreak.BreakType.EOL_SURE_SPACE,
                                                vision.TextAnnotation.DetectedBreak.BreakType.LINE_BREAK):
                                    flush_line()
                                elif brk_type in (vision.TextAnnotation.DetectedBreak.BreakType.SPACE,
                                                  vision.TextAnnotation.DetectedBreak.BreakType.SURE_SPACE):
                                    current_line_text.append(' ')
                            # Add a space between words if no symbol-level break indicated
                            current_line_text.append(' ')
                        # End of paragraph: flush residual line
                        flush_line()

            self.last_full_text = '\n'.join([p for p in full_text_parts if p])

            # Draw rectangles for each region
            for r in regions:
                cv2.rectangle(frame_bgr, (r["x"], r["y"]), (r["x"] + r["w"], r["y"] + r["h"]), (0, 128, 255), 2)

            return frame_bgr, regions
        except Exception as e:
            print(f"Error in Google Vision OCR: {e}")
            return frame_bgr, []
    def detect_text_fallback(self, frame):
        """Fast, lightweight contour-based text region detection (no OCR)."""
        try:
            frame_h, frame_w = frame.shape[:2]
            # Downscale for speed if large, keep scale to map back
            scale = 0.5 if max(frame_w, frame_h) > 700 else 1.0
            if scale != 1.0:
                small = cv2.resize(frame, (int(frame_w * scale), int(frame_h * scale)))
            else:
                small = frame

            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (3, 3), 0)

            # Gradients emphasize strokes
            grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
            grad = cv2.convertScaleAbs(cv2.absdiff(grad_x, grad_y))
            _, thresh = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
            closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
            closed = cv2.dilate(closed, kernel, iterations=1)

            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            regions = []
            for c in contours:
                x, y, w, h = cv2.boundingRect(c)
                # Map back to original scale
                if scale != 1.0:
                    x = int(x / scale)
                    y = int(y / scale)
                    w = int(w / scale)
                    h = int(h / scale)

                if self.is_valid_text_region(w, h) and self.is_in_center_region(
                    x, y, w, h, frame_w, frame_h
                ):
                    regions.append((x, y, w, h))

            regions = self.remove_overlapping_regions(regions, overlap_threshold=0.5)

            # Update last regions with a simple score = area
            self.last_text_regions = [
                {"x": x, "y": y, "w": w, "h": h, "score": float(w*h), "label": ""}
                for (x, y, w, h) in regions
            ]

            for (x, y, w, h) in regions:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            return frame, len(regions)

        except Exception as e:
            print(f"Error in fallback detection: {e}")
            return frame, 0
    
    def start_camera(self):
        """Initialize and start the camera with error handling - tries to find iPhone camera first"""
        try:
            # Try to find iPhone camera (usually camera 1 or 2)
            camera_index = 0
            best_camera = None
            best_fps = 0
            
            print("Scanning for available cameras...")
            
            # Test cameras 0, 1, 2 to find the best one (iPhone usually has higher FPS)
            for i in range(3):
                test_camera = cv2.VideoCapture(i)
                if test_camera.isOpened():
                    # Set high resolution and FPS
                    test_camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                    test_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                    test_camera.set(cv2.CAP_PROP_FPS, 60)  # Try 60 FPS for iPhone
                    
                    # Get actual properties
                    actual_fps = test_camera.get(cv2.CAP_PROP_FPS)
                    actual_width = test_camera.get(cv2.CAP_PROP_FRAME_WIDTH)
                    actual_height = test_camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    
                    print(f"Camera {i}: {actual_width:.0f}x{actual_height:.0f} @ {actual_fps:.1f} FPS")
                    
                    # Prefer camera with higher FPS (likely iPhone)
                    if actual_fps > best_fps:
                        best_fps = actual_fps
                        best_camera = i
                        if actual_fps >= 30:  # Found a good camera
                            break
                    
                    test_camera.release()
            
            # Use the best camera found
            if best_camera is not None:
                camera_index = best_camera
                print(f"Selected camera {camera_index} with {best_fps:.1f} FPS")
            else:
                print("No suitable camera found, using default camera 0")
                camera_index = 0
            
            # Open the selected camera
            self.camera = cv2.VideoCapture(camera_index)
            
            if not self.camera.isOpened():
                print(f"Error: Could not open camera {camera_index}. Please check if your camera is connected.")
                return False
            
            # Set camera properties for best performance
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            self.camera.set(cv2.CAP_PROP_FPS, 60)  # Try 60 FPS
            
            # Prefer MJPG to reduce decode overhead on many webcams
            try:
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                self.camera.set(cv2.CAP_PROP_FOURCC, fourcc)
            except Exception:
                pass
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer for lower latency
            
            # Test camera by reading a frame
            ret, test_frame = self.camera.read()
            if not ret:
                print("Error: Camera opened but cannot read frames.")
                return False
            
            actual_w = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.camera.get(cv2.CAP_PROP_FPS)
            print(f"Camera initialized: {actual_w}x{actual_h} @ {actual_fps:.1f} FPS")
            return True
            
        except Exception as e:
            print(f"Error initializing camera: {e}")
            return False
    
    def detect_text_easyocr(self, frame):
        """Detect text using EasyOCR with improved performance"""
        try:
            # Convert BGR to RGB for EasyOCR
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize frame for faster processing
            height, width = rgb_frame.shape[:2]
            if width > 800:  # Resize if too large
                scale = 800 / width
                new_width = int(width * scale)
                new_height = int(height * scale)
                rgb_frame = cv2.resize(rgb_frame, (new_width, new_height))
            
            # Detect text with optimized parameters for speed
            results = self.ocr_reader.readtext(
                rgb_frame,
                paragraph=False,
                width_ths=0.7,
                height_ths=0.7,
                mag_ratio=1.5,
                low_text=0.3,
            )
            
            text_count = 0
            detected_regions = []
            # Draw bounding boxes around detected text
            for (bbox, text, confidence) in results:
                # Higher confidence threshold for better accuracy
                if confidence > 0.35 and len(text.strip()) > 0:
                    # Extract bounding box coordinates
                    (tl, tr, br, bl) = bbox
                    tl = (int(tl[0]), int(tl[1]))
                    br = (int(br[0]), int(br[1]))
                    
                    # Scale back to original size if resized
                    if width > 800:
                        scale_factor = width / 800
                        tl = (int(tl[0] * scale_factor), int(tl[1] * scale_factor))
                        br = (int(br[0] * scale_factor), int(br[1] * scale_factor))
                    
                    # Draw blue rectangle around text
                    cv2.rectangle(frame, tl, br, (255, 0, 0), 2)
                    
                    # Add text label with confidence
                    label = f"{text[:20]} ({confidence:.2f})"
                    cv2.putText(frame, label, (tl[0], tl[1] - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    
                    # Track region
                    x, y = tl
                    w, h = (br[0] - tl[0]), (br[1] - tl[1])
                    detected_regions.append({
                        "x": x, "y": y, "w": w, "h": h, "score": float(confidence),
                        "label": text
                    })
                    text_count += 1

            # Persist regions
            self.last_text_regions = detected_regions
            
            return frame, text_count
            
        except Exception as e:
            print(f"Error in text detection: {e}")
            return frame, 0
    
    def detect_text_tesseract(self, frame):
        """Detect text using Tesseract OCR"""
        try:
            import pytesseract
            
            # Convert to grayscale for better OCR
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Get text data with bounding boxes
            data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
            
            detected_regions = []
            # Draw bounding boxes around detected text
            text_count = 0
            for i, conf in enumerate(data['conf']):
                if conf > 30:  # Lower confidence threshold
                    x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                    text = data['text'][i].strip()
                    
                    if text and len(text) > 0:  # Only draw boxes for non-empty text
                        # Draw blue rectangle
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                        
                        # Add text label
                        label = f"{text[:15]} ({conf}%)"
                        cv2.putText(frame, label, (x, y - 5), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                        detected_regions.append({
                            "x": int(x), "y": int(y), "w": int(w), "h": int(h),
                            "score": float(conf) / 100.0, "label": text
                        })
                        text_count += 1
                        
            self.last_text_regions = detected_regions
            return frame, text_count
            
        except Exception as e:
            print(f"Error in Tesseract detection: {e}")
            return frame, 0
    
    def detect_text_accurate(self, frame):
        """Accurate text detection using EasyOCR with advanced filtering - optimized for performance"""
        try:
            # Convert BGR to RGB for EasyOCR
            # Light contrast enhancement (CLAHE) on luminance, then back to RGB
            bgr = frame
            lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            lab = cv2.merge((cl, a, b))
            enhanced_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            rgb_frame = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
            orig_h, orig_w = rgb_frame.shape[:2]
            scaled = False
            target_max_w = 1200  # Higher scale to pick up small text
            if orig_w > target_max_w:
                scale = target_max_w / float(orig_w)
                new_w = int(orig_w * scale)
                new_h = int(orig_h * scale)
                rgb_proc = cv2.resize(rgb_frame, (new_w, new_h))
                scaled = True
            else:
                rgb_proc = rgb_frame
            
            # Detect text with EasyOCR (optimized for accuracy and speed)
            results = self.ocr_reader.readtext(
                rgb_proc,
                paragraph=False,
                width_ths=0.85,
                height_ths=0.85,
                text_threshold=0.6,
                link_threshold=0.4,
                low_text=0.3,
            )

            # If nothing found, retry once at an even higher scale & more permissive settings
            if not results:
                target_max_w2 = 1600
                if orig_w < target_max_w2:
                    scale2 = min(1.6, target_max_w2 / float(orig_w))
                    rgb_proc2 = cv2.resize(rgb_frame, (int(orig_w * scale2), int(orig_h * scale2)))
                    scaled2 = True
                else:
                    rgb_proc2 = rgb_frame
                    scaled2 = False
                results = self.ocr_reader.readtext(
                    rgb_proc2,
                    paragraph=False,
                    width_ths=0.9,
                    height_ths=0.9,
                    text_threshold=0.55,
                    link_threshold=0.4,
                    low_text=0.25,
                )
            
            text_regions = []
            frame_height, frame_width = frame.shape[:2]
            
            # Process each detected text region
            for (bbox, text, confidence) in results:
                # High confidence threshold for accuracy
                if confidence > 0.5 and len(text.strip()) > 0:
                    # Extract bounding box coordinates
                    (tl, tr, br, bl) = bbox
                    if scaled:
                        scale_back = orig_w / float(rgb_proc.shape[1])
                        tl = (int(tl[0] * scale_back), int(tl[1] * scale_back))
                        br = (int(br[0] * scale_back), int(br[1] * scale_back))
                    elif 'rgb_proc2' in locals() and scaled2:
                        scale_back = orig_w / float(rgb_proc2.shape[1])
                        tl = (int(tl[0] * scale_back), int(tl[1] * scale_back))
                        br = (int(br[0] * scale_back), int(br[1] * scale_back))
                    else:
                        tl = (int(tl[0]), int(tl[1]))
                        br = (int(br[0]), int(br[1]))
                    
                    # Calculate width and height
                    w = br[0] - tl[0]
                    h = br[1] - tl[1]
                    
                    # Add margin to the bounding box for better capture
                    margin_x = max(10, int(w * 0.2))  # 20% margin or minimum 10px
                    margin_y = max(8, int(h * 0.3))   # 30% margin or minimum 8px
                    
                    # Expand the bounding box with margins
                    x = max(0, tl[0] - margin_x)
                    y = max(0, tl[1] - margin_y)
                    w = min(frame_width - x, w + 2 * margin_x)
                    h = min(frame_height - y, h + 2 * margin_y)
                    
                    # Validate text content, region size, and center position
                    if (self.is_valid_text_content(text) and 
                        self.is_valid_text_region(w, h) and
                        self.is_in_center_region(x, y, w, h, frame_width, frame_height)):
                        text_regions.append((x, y, w, h, text, confidence))
            
            # Remove overlapping regions
            text_regions = self.remove_overlapping_text_regions(text_regions)
            
            # Persist last regions
            self.last_text_regions = [
                {"x": x, "y": y, "w": w, "h": h, "score": float(conf), "label": text}
                for (x, y, w, h, text, conf) in text_regions
            ]

            # Draw bounding boxes
            for i, (x, y, w, h, text, conf) in enumerate(text_regions):
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                # Show text and confidence
                label = f"{text[:20]} ({conf:.2f})"
                cv2.putText(frame, label, (x, y - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            return frame, len(text_regions)
            
        except Exception as e:
            print(f"Error in accurate text detection: {e}")
            return frame, 0
    
    def is_in_center_region(self, x, y, w, h, frame_width, frame_height):
        """Check if text region is in the left half of the screen (OCR reading area)"""
        # For split-screen layout, we want text in the left half
        mid_x = frame_width // 2
        
        # Text region boundaries
        text_left = x
        text_right = x + w
        text_top = y
        text_bottom = y + h
        
        # Check if text region is mostly within left half
        # At least 50% of the text should be in the left region
        left_overlap_x = max(0, min(text_right, mid_x) - max(text_left, 0))
        left_overlap_y = max(0, min(text_bottom, frame_height) - max(text_top, 0))
        
        text_area = w * h
        left_overlap_area = left_overlap_x * left_overlap_y
        
        # Text is valid if at least 50% is in left region
        return left_overlap_area >= text_area * 0.5

    def _compute_regions_signature(self, regions: List[dict], frame_w: int, frame_h: int) -> str:
        """Build a stable signature for currently detected left-side regions to detect changes.
        Uses labels and quantized geometry to reduce jitter - much more tolerant to minor movements.
        """
        if not regions:
            return ""
        sig_items = []
        for r in regions:
            x, y, w, h = r.get("x", 0), r.get("y", 0), r.get("w", 0), r.get("h", 0)
            if not self.is_in_center_region(x, y, w, h, frame_w, frame_h):
                continue
            label = str(r.get("label", "")).strip()
            # Quantize to much larger grid to be very tolerant to minor movements
            q = 50  # Increased from 8 to 50 pixels for much more tolerance
            qx, qy, qw, qh = x // q, y // q, max(1, w // q), max(1, h // q)
            sig_items.append(f"{label}|{qx},{qy},{qw},{qh}")
        if not sig_items:
            return ""
        sig_items.sort()
        return "#".join(sig_items)

    def _start_autosolve(self, regions: List[dict], frame_for_crop: np.ndarray):
        if self._autosolve_in_progress:
            return
        
        # Ensure we capture complete equations by clustering related text blocks
        complete_regions = self._ensure_complete_equation(regions, frame_for_crop)
        
        # Prepare parts similar to key 'c'
        parts: List[dict] = []
        # Send complete equation regions instead of individual text blocks
        for r in complete_regions:
            x, y, w, h = r["x"], r["y"], r["w"], r["h"]
            x0 = max(0, x)
            y0 = max(0, y)
            x1 = min(frame_for_crop.shape[1], x + w)
            y1 = min(frame_for_crop.shape[0], y + h)
            if x1 <= x0 or y1 <= y0:
                continue
            crop = frame_for_crop[y0:y1, x0:x1]
            crop = self.preprocess_crop_for_ocr(crop)
            parts.append({"img": crop, "text": r.get("label", "")})
        if not parts:
            return

        def worker():
            try:
                self._autosolve_in_progress = True
                answer = self.send_multiple_images_to_llm(parts)
                if answer:
                    concise = self._extract_concise_answer(answer)
                    print("\n=== LLM Solution (autosolve) ===\n" + answer + "\n===============================\n")
                    if concise:
                        print(f"Answer: {concise}")
                    # Persist full answer for on-screen overlay
                    self.last_llm_solution_text = answer.strip()
                    self.last_llm_solution_time = time.time()
                else:
                    print("Autosolve: LLM did not return a solution.")
            finally:
                # Reset so user can autosolve again after changes
                self._autosolve_in_progress = False
                self._hold_signature = None
                self._hold_start_time = 0.0

        threading.Thread(target=worker, daemon=True).start()
    
    def is_valid_text_region(self, w, h):
        """Check if a region is likely to contain text"""
        # Size constraints (broadened further to catch small words)
        if w < 12 or h < 8 or w > 1600 or h > 500:
            return False
        
        # Aspect ratio constraints
        aspect_ratio = w / float(h)
        if aspect_ratio < 0.15 or aspect_ratio > 30:
            return False
        
        # Area constraints (updated for larger regions)
        area = w * h
        if area < 80 or area > 600000:
            return False
        
        return True

    def is_valid_word_region(self, w: int, h: int) -> bool:
        """Looser constraints for tiny word tokens (e.g., subscripts)."""
        if w < 6 or h < 6 or w > 2000 or h > 600:
            return False
        aspect_ratio = w / float(max(1, h))
        if aspect_ratio < 0.08 or aspect_ratio > 50:
            return False
        area = w * h
        if area < 30 or area > 800000:
            return False
        return True
    
    def is_valid_text_content(self, text):
        """Check if the text content is valid (contains actual readable characters)"""
        # Remove common OCR artifacts
        text = text.strip()
        
        # Must have minimum length
        if len(text) < 1:
            return False
        
        # Check if it contains actual readable characters (not just symbols)
        readable_chars = sum(1 for c in text if c.isalnum())
        if readable_chars < 1:
            return False
        
        # Check if it's not just repeated characters (but allow short repeated text)
        if len(set(text)) < 2 and len(text) > 3:
            return False
        
        # Check if it's not just punctuation (but allow some punctuation)
        if all(not c.isalnum() for c in text) and len(text) > 2:
            return False
        
        # Allow common text patterns
        if len(text) >= 1 and readable_chars >= 1:
            return True
        
        return False
    
    def remove_overlapping_regions(self, regions, overlap_threshold=0.5):
        """Remove overlapping text regions"""
        if not regions:
            return regions
        
        # Sort regions by area (largest first)
        regions = sorted(regions, key=lambda x: x[2] * x[3], reverse=True)
        
        filtered_regions = []
        for region in regions:
            x1, y1, w1, h1 = region
            
            # Check overlap with existing regions
            is_overlapping = False
            for existing in filtered_regions:
                x2, y2, w2, h2 = existing
                
                # Calculate intersection
                x_left = max(x1, x2)
                y_top = max(y1, y2)
                x_right = min(x1 + w1, x2 + w2)
                y_bottom = min(y1 + h1, y2 + h2)
                
                if x_right > x_left and y_bottom > y_top:
                    intersection = (x_right - x_left) * (y_bottom - y_top)
                    area1 = w1 * h1
                    area2 = w2 * h2
                    smaller_area = min(area1, area2)
                    
                    if intersection / smaller_area > overlap_threshold:
                        is_overlapping = True
                        break
            
            if not is_overlapping:
                filtered_regions.append(region)
        
        return filtered_regions
    
    def remove_overlapping_text_regions(self, regions, overlap_threshold=0.5):
        """Remove overlapping text regions, keeping the one with higher confidence"""
        if not regions:
            return regions
        
        # Sort regions by confidence (highest first)
        regions = sorted(regions, key=lambda x: x[5], reverse=True)
        
        filtered_regions = []
        for region in regions:
            x1, y1, w1, h1, text1, conf1 = region
            
            # Check overlap with existing regions
            is_overlapping = False
            for existing in filtered_regions:
                x2, y2, w2, h2, text2, conf2 = existing
                
                # Calculate intersection
                x_left = max(x1, x2)
                y_top = max(y1, y2)
                x_right = min(x1 + w1, x2 + w2)
                y_bottom = min(y1 + h1, y2 + h2)
                
                if x_right > x_left and y_bottom > y_top:
                    intersection = (x_right - x_left) * (y_bottom - y_top)
                    area1 = w1 * h1
                    area2 = w2 * h2
                    smaller_area = min(area1, area2)
                    
                    if intersection / smaller_area > overlap_threshold:
                        is_overlapping = True
                        break
            
            if not is_overlapping:
                filtered_regions.append(region)
        
        return filtered_regions
    
    def detect_text(self, frame):
        """Detect text in the frame using available OCR method"""
        if not self.detection_enabled:
            return frame, 0
        
        if self.ocr_reader is None:
            return frame, 0
        
        # Prefer Google Vision if available for robust small-text OCR
        if self.vision_client is not None:
            annotated, regions = self.google_vision_ocr(frame.copy())
            if regions:
                self.last_text_regions = regions
                return annotated, len(regions)

        # Prefer accurate OCR when available
        if ('easyocr' in sys.modules) and isinstance(self.ocr_reader, easyocr.Reader):
            if self.accurate_mode:
                return self.detect_text_accurate(frame)
            else:
                return self.detect_text_easyocr(frame)
        # Fallbacks
        if self.ocr_reader == 'tesseract':
            return self.detect_text_tesseract(frame)
        return self.detect_text_fallback(frame)

    def _detection_worker(self):
        """Background worker that runs OCR on most recent frames without blocking UI."""
        while not self._det_stop.is_set():
            try:
                frame = self._det_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            try:
                # Enforce interval to avoid overwork
                now = time.time()
                if now - self._det_last_time < self._det_interval_sec:
                    continue
                self._det_last_time = now
                # Run detection using current mode; updates self.last_text_regions internally
                _disp, _cnt = self.detect_text(frame)
            except Exception as e:
                print(f"Detection worker error: {e}")
            finally:
                # Mark task done; if new frames arrived, queue will retain only last due to maxsize=1 logic
                pass

    def _start_detection_thread(self):
        if self._det_thread is None or not self._det_thread.is_alive():
            self._det_stop.clear()
            self._det_thread = threading.Thread(target=self._detection_worker, daemon=True)
            self._det_thread.start()

    def _stop_detection_thread(self):
        try:
            self._det_stop.set()
            if self._det_thread is not None:
                self._det_thread.join(timeout=0.5)
        except Exception:
            pass

    def _offer_frame_to_detector(self, frame: np.ndarray):
        """Try to push frame to detector without blocking; drop older frame if needed."""
        try:
            if self._det_queue.full():
                try:
                    _ = self._det_queue.get_nowait()
                except queue.Empty:
                    pass
            self._det_queue.put_nowait(frame)
        except Exception:
            pass

    def _draw_regions(self, frame: np.ndarray, regions: List[dict]):
        for r in regions:
            x, y, w, h = r["x"], r["y"], r["w"], r["h"]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            label = r.get("label")
            if label:
                txt = str(label)[:20]
                cv2.putText(frame, txt, (x, max(10, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    def select_best_region(self) -> Optional[dict]:
        """Select the best detected region using score and area heuristic."""
        if not self.last_text_regions:
            return None
        # Compute composite score = normalized area * 0.5 + score * 0.5
        areas = [r["w"] * r["h"] for r in self.last_text_regions]
        max_area = max(areas) if areas else 1.0
        def composite(r):
            area_component = (r["w"] * r["h"]) / max_area
            score_component = float(r.get("score", 0.0))
            return 0.5 * area_component + 0.5 * score_component
        return max(self.last_text_regions, key=composite)

    def _cluster_blocks(self, blocks: List[dict], frame_w: int, frame_h: int) -> List[List[dict]]:
        """Cluster text blocks by spatial proximity and overlap to form problem-level groups."""
        if not blocks:
            return []
        # Simple agglomerative clustering using IoU/neighbor distance
        clusters: List[List[dict]] = []
        def iou(a, b):
            ax0, ay0, ax1, ay1 = a["x"], a["y"], a["x"] + a["w"], a["y"] + a["h"]
            bx0, by0, bx1, by1 = b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]
            ix0, iy0 = max(ax0, bx0), max(ay0, by0)
            ix1, iy1 = min(ax1, bx1), min(ay1, by1)
            iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
            inter = iw * ih
            area_a = (ax1 - ax0) * (ay1 - ay0)
            area_b = (bx1 - bx0) * (by1 - by0)
            union = area_a + area_b - inter + 1e-6
            return inter / union
        def are_neighbors(a, b):
            # Consider neighbors if overlapping (IoU) or within small gap vertically/horizontally
            if iou(a, b) > 0.02:
                return True
            # Horizontal/vertical proximity thresholds relative to avg height
            avg_h = (a["h"] + b["h"]) / 2.0
            avg_w = (a["w"] + b["w"]) / 2.0
            gap_x = max(0, max(a["x"], b["x"]) - min(a["x"] + a["w"], b["x"] + b["w"]))
            gap_y = max(0, max(a["y"], b["y"]) - min(a["y"] + a["h"], b["y"] + b["h"]))
            return gap_x < avg_w * 0.3 or gap_y < avg_h * 0.6
        for block in blocks:
            placed = False
            for cluster in clusters:
                if any(are_neighbors(block, other) for other in cluster):
                    cluster.append(block)
                    placed = True
                    break
            if not placed:
                clusters.append([block])
        # Merge clusters that became adjacent
        merged = True
        while merged and len(clusters) > 1:
            merged = False
            out = []
            while clusters:
                c = clusters.pop()
                merged_into_existing = False
                for d in out:
                    if any(are_neighbors(a, b) for a in c for b in d):
                        d.extend(c)
                        merged_into_existing = True
                        merged = True
                        break
                if not merged_into_existing:
                    out.append(c)
            clusters = out
        return clusters

    def _ensure_complete_equation(self, regions: List[dict], frame_for_crop: np.ndarray) -> List[dict]:
        """Ensure we capture the complete equation by expanding the detection area and clustering related text."""
        if not regions:
            return regions
        
        # Cluster nearby text blocks to form complete equations
        clusters = self._cluster_blocks(regions, frame_for_crop.shape[1], frame_for_crop.shape[0])
        
        # For each cluster, create a bounding box that encompasses all text
        complete_regions = []
        for cluster in clusters:
            if not cluster:
                continue
            
            # Calculate the bounding box that encompasses all text in the cluster
            min_x = min(block["x"] for block in cluster)
            min_y = min(block["y"] for block in cluster)
            max_x = max(block["x"] + block["w"] for block in cluster)
            max_y = max(block["y"] + block["h"] for block in cluster)
            
            # Add padding to ensure we capture the complete equation
            padding_x = int(max(20, (max_x - min_x) * 0.1))  # 10% padding or minimum 20px
            padding_y = int(max(15, (max_y - min_y) * 0.15))  # 15% padding or minimum 15px
            
            # Ensure we don't go outside frame boundaries
            x = max(0, min_x - padding_x)
            y = max(0, min_y - padding_y)
            w = min(frame_for_crop.shape[1] - x, max_x - min_x + 2 * padding_x)
            h = min(frame_for_crop.shape[0] - y, max_y - min_y + 2 * padding_y)
            
            # Combine all text labels from the cluster
            combined_text = " ".join(block.get("label", "") for block in cluster if block.get("label"))
            
            # Create a single region representing the complete equation
            complete_region = {
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "score": max(block.get("score", 0.9) for block in cluster),
                "label": combined_text,
                "is_complete_equation": True
            }
            
            complete_regions.append(complete_region)
        
        return complete_regions if complete_regions else regions

    def preprocess_crop_for_ocr(self, crop_bgr: np.ndarray) -> np.ndarray:
        """Enhanced preprocessing for small text and math symbols: upscale, CLAHE, denoise, and unsharp mask."""
        img = crop_bgr.copy()
        h, w = img.shape[:2]
        
        # Upscale to improve readability (cap to 2000px on longer side for better math symbol recognition)
        max_side = max(h, w)
        target_side = max(1200, min(2000, int(max_side * 2.0)))  # Increased scale factor
        scale = target_side / float(max_side)
        if scale > 1.05:
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
        
        # Enhanced CLAHE on luminance with better parameters for math symbols
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))  # Increased clip limit
        cl = clahe.apply(l)
        lab = cv2.merge((cl, a, b))
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Mild denoising while preserving edges (important for math symbols)
        img = cv2.bilateralFilter(img, d=5, sigmaColor=75, sigmaSpace=75)
        
        # Enhanced unsharp mask for better text crispness
        blur = cv2.GaussianBlur(img, (0, 0), sigmaX=1.5)
        img = cv2.addWeighted(img, 1.8, blur, -0.8, 0)  # Increased sharpening
        
        # Additional contrast enhancement for math symbols
        img = cv2.convertScaleAbs(img, alpha=1.1, beta=5)  # Slight contrast boost
        
        return img

    def preprocess_for_vision(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Preprocess frame for Google Vision without distorting geometry: CLAHE, mild denoise, unsharp."""
        bgr = frame_bgr
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        lab = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        # Mild denoise while preserving edges
        denoised = cv2.bilateralFilter(enhanced, d=5, sigmaColor=60, sigmaSpace=60)
        # Unsharp mask for text crispness
        blur = cv2.GaussianBlur(denoised, (0, 0), sigmaX=1.2)
        sharp = cv2.addWeighted(denoised, 1.6, blur, -0.6, 0)
        return sharp

    def normalize_math_text(self, text: str) -> str:
        """Normalize common OCR confusions for math text (≥, ≤, minus, quotes, spacing)."""
        if not text:
            return ""
        replacements = {
            '≥': '>=',
            '≤': '<=',
            '≠': '!=',
            '−': '-',
            '–': '-',
            '—': '-',
            '“': '"',
            '”': '"',
            "’": "'",
            "‘": "'",
            '×': 'x',  # times sign -> x (common in algebra text)
        }
        out = text
        for k, v in replacements.items():
            out = out.replace(k, v)
        # Collapse excessive spaces
        out = re.sub(r"\s+", " ", out)
        # Join variable + subscript digit patterns like 'x 1' -> 'x1'
        out = re.sub(r"\b([a-zA-Z])\s+(\d)\b", r"\1\2", out)
        # Ensure spaces around +, -, = for readability unless already there
        out = re.sub(r"\s*([+=\-])\s*", r" \1 ", out)
        out = re.sub(r"\s+", " ", out).strip()
        return out

    def merge_subscript_word_boxes(self, boxes: List[dict]) -> List[dict]:
        """Merge adjacent word boxes that look like a base symbol with a subscript (e.g., 'x' with lower '1').
        The merged label is concatenated without space (e.g., 'x' + '1' -> 'x1').
        """
        if not boxes:
            return boxes

        # Work on a copy, sort by x then y
        boxes = [dict(b) for b in boxes]
        boxes.sort(key=lambda b: (b["x"], b["y"]))

        def can_merge_as_subscript(left: dict, right: dict) -> bool:
            left_label = str(left.get("label", ""))
            right_label = str(right.get("label", ""))
            if not left_label or not right_label:
                return False
            # Subscript candidate should be short (1-2 chars), typically digits or single letter
            if len(right_label) > 2:
                return False
            if not re.match(r"^[0-9a-zA-Z]+$", right_label):
                return False
            # Left should be a variable-like token (letters possibly with digits already)
            if not re.match(r"^[a-zA-Z][a-zA-Z0-9]*$", left_label):
                return False
            xi, yi, wi, hi = left["x"], left["y"], left["w"], left["h"]
            xj, yj, wj, hj = right["x"], right["y"], right["w"], right["h"]
            # Right should be horizontally adjacent (small gap) and vertically lower and smaller height
            gap_x = max(0, xj - (xi + wi))
            if gap_x > max(6, 0.25 * wi):
                return False
            yc_i = yi + hi / 2.0
            yc_j = yj + hj / 2.0
            # Subscript sits below baseline: center lower by at least ~12% of left height
            if (yc_j - yc_i) < 0.12 * hi:
                return False
            # Height should be smaller (typical subscripts)
            if hj >= 0.95 * hi:
                return False
            # Vertical overlap shouldn't be zero (avoid completely separate lines)
            overlap_y = min(yi + hi, yj + hj) - max(yi, yj)
            if overlap_y <= 0:
                return False
            return True

        merged_something = True
        while merged_something:
            merged_something = False
            used = [False] * len(boxes)
            new_boxes: List[dict] = []
            i = 0
            while i < len(boxes):
                if used[i]:
                    i += 1
                    continue
                left = boxes[i]
                # Look ahead to the immediate neighbor on the right
                j = i + 1
                merged = False
                if j < len(boxes) and not used[j]:
                    right = boxes[j]
                    if can_merge_as_subscript(left, right):
                        # Merge into one box
                        x0 = min(left["x"], right["x"])
                        y0 = min(left["y"], right["y"])
                        x1 = max(left["x"] + left["w"], right["x"] + right["w"])
                        y1 = max(left["y"] + left["h"], right["y"] + right["h"])
                        merged_box = {
                            "x": x0,
                            "y": y0,
                            "w": x1 - x0,
                            "h": y1 - y0,
                            "score": float((left.get("score", 0.9) + right.get("score", 0.9)) / 2.0),
                            "label": f"{left.get('label','')}{right.get('label','')}"
                        }
                        new_boxes.append(merged_box)
                        used[i] = True
                        used[j] = True
                        merged = True
                        merged_something = True
                        i = j + 1
                if not merged:
                    new_boxes.append(left)
                    used[i] = True
                    i += 1
            boxes = new_boxes

        return boxes

    def ocr_extract_text(self, crop_bgr: np.ndarray) -> str:
        """Extract text string from the cropped region using available OCR."""
        try:
            txt = ""
            # ALWAYS try Mathpix first for mathematical content if configured
            if self.mathpix_enabled:
                try:
                    # Enhanced preprocessing for Mathpix
                    crop_enhanced = self.preprocess_crop_for_ocr(crop_bgr)
                    success, buf = cv2.imencode('.png', crop_enhanced)
                    if success:
                        b64 = base64.b64encode(buf.tobytes()).decode('utf-8')
                        headers = {
                            'app_id': self.mathpix_app_id,
                            'app_key': self.mathpix_app_key,
                            'Content-type': 'application/json'
                        }
                        # Enhanced Mathpix request with better options
                        payload = {
                            'src': f'data:image/png;base64,{b64}',
                            'formats': ['text', 'latex_simplified', 'asciimath'],
                            'data_options': {
                                'include_asciimath': True, 
                                'include_latex': True,
                                'include_line_data': True,
                                'include_confidence': True
                            },
                            'ocr': ['math', 'text'],
                            'math_inline_delimiters': ['$', '$'],
                            'math_display_delimiters': ['$$', '$$'],
                            'rm_spaces': True,
                            'math_rm_spaces': True
                        }
                        resp = requests.post('https://api.mathpix.com/v3/text', headers=headers, json=payload, timeout=15)
                        if resp.ok:
                            data = resp.json()
                            # Prioritize text output, then latex_simplified, then asciimath
                            if isinstance(data, dict):
                                if data.get('text') and data['text'].strip():
                                    result = self.normalize_math_text(data['text'].strip())
                                    if result:
                                        return result
                                if data.get('latex_simplified') and data['latex_simplified'].strip():
                                    result = data['latex_simplified'].strip()
                                    if result:
                                        return result
                                if data.get('asciimath') and data['asciimath'].strip():
                                    result = data['asciimath'].strip()
                                    if result:
                                        return result
                        else:
                            print(f"Mathpix API error: {resp.status_code} - {resp.text}")
                except Exception as e:
                    print(f"Mathpix OCR failed: {e}")
            
            # Fallback to Google Vision with enhanced preprocessing
            if self.vision_client is not None:
                try:
                    from google.cloud import vision
                    # Enhanced preprocessing for Google Vision
                    crop_proc = self.preprocess_for_vision(crop_bgr)
                    success, buf = cv2.imencode('.png', crop_proc)
                    if success:
                        image = vision.Image(content=buf.tobytes())
                        image_context = vision.ImageContext(
                            language_hints=['en'], 
                            text_detection_params=vision.TextDetectionParams(enable_text_detection_confidence_score=True)
                        )
                        resp = self.vision_client.document_text_detection(image=image, image_context=image_context)
                        if resp.error.message:
                            print(f"Google Vision OCR error: {resp.error.message}")
                        else:
                            if resp.full_text_annotation and resp.full_text_annotation.text:
                                result = self.normalize_math_text(resp.full_text_annotation.text.strip())
                                if result:
                                    return result
                except Exception as e:
                    print(f"Google Vision crop OCR failed: {e}")
            
            # Final fallback to EasyOCR with enhanced preprocessing
            if ('easyocr' in sys.modules) and isinstance(self.ocr_reader, easyocr.Reader):
                try:
                    # Enhanced preprocessing for EasyOCR
                    crop_enhanced = self.preprocess_crop_for_ocr(crop_bgr)
                    rgb = cv2.cvtColor(crop_enhanced, cv2.COLOR_BGR2RGB)
                    results = self.ocr_reader.readtext(
                        rgb, 
                        paragraph=True,
                        width_ths=0.9,
                        height_ths=0.9,
                        text_threshold=0.6,
                        link_threshold=0.4,
                        low_text=0.3
                    )
                    parts = []
                    for (bbox, text, conf) in results:
                        if text and conf > 0.4:  # Higher confidence threshold
                            normalized = self.normalize_math_text(text)
                            if normalized:
                                parts.append(normalized)
                    txt = "\n".join(parts).strip()
                    if txt:
                        return txt
                except Exception as e:
                    print(f"EasyOCR fallback failed: {e}")
            
            # Last resort: Tesseract
            try:
                import pytesseract
                crop_enhanced = self.preprocess_crop_for_ocr(crop_bgr)
                gray = cv2.cvtColor(crop_enhanced, cv2.COLOR_BGR2GRAY)
                txt = self.normalize_math_text(pytesseract.image_to_string(gray).strip())
                if txt:
                    return txt
            except Exception as e:
                print(f"Tesseract fallback failed: {e}")
            
            return ""
        except Exception as e:
            print(f"OCR extraction failed: {e}")
            return ""

    def save_crop(self, crop_bgr: np.ndarray) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ocr_crop_{timestamp}.jpg"
        filepath = os.path.join(self.output_dir, filename)
        cv2.imwrite(filepath, crop_bgr)
        return filepath

    def _extract_concise_answer(self, text: str) -> str:
        """Try to extract a concise answer from model output."""
        if not text:
            return ""
        lower = text.lower()
        for key in ["answer:", "final answer:", "result:"]:
            if key in lower:
                idx = lower.rfind(key)
                return text[idx + len(key):].strip().splitlines()[0].strip()
        lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
        if lines:
            return lines[-1]
        return text.strip()

    def _clean_solution_text(self, text: str) -> str:
        """Clean up solution text to be more readable (remove LaTeX formatting and structure neatly)."""
        # Remove LaTeX math delimiters
        text = re.sub(r'\\\[(.*?)\\\]', r'\1', text)  # \[ ... \] -> ...
        text = re.sub(r'\\\((.*?)\\\)', r'\1', text)  # \( ... \) -> ...
        
        # Replace common LaTeX symbols with readable equivalents
        replacements = {
            r'\\frac\{([^}]+)\}\{([^}]+)\}': r'\1/\2',  # \frac{a}{b} -> a/b
            r'\\sqrt\{([^}]+)\}': r'√\1',  # \sqrt{x} -> √x
            r'\\times': '×',  # \times -> ×
            r'\\div': '÷',  # \div -> ÷
            r'\\geq': '≥',  # \geq -> ≥
            r'\\leq': '≤',  # \leq -> ≤
            r'\\neq': '≠',  # \neq -> ≠
            r'\\pm': '±',  # \pm -> ±
            r'\\infty': '∞',  # \infty -> ∞
            r'\\sum': 'Σ',  # \sum -> Σ
            r'\\int': '∫',  # \int -> ∫
            r'\\alpha': 'α',  # \alpha -> α
            r'\\beta': 'β',  # \beta -> β
            r'\\gamma': 'γ',  # \gamma -> γ
            r'\\delta': 'δ',  # \delta -> δ
            r'\\theta': 'θ',  # \theta -> θ
            r'\\pi': 'π',  # \pi -> π
            r'\\sigma': 'σ',  # \sigma -> σ
            r'\\mu': 'μ',  # \mu -> μ
            r'\\lambda': 'λ',  # \lambda -> λ
            r'\\cdot': '·',  # \cdot -> ·
            r'\\approx': '≈',  # \approx -> ≈
            r'\\propto': '∝',  # \propto -> ∝
            r'\\partial': '∂',  # \partial -> ∂
            r'\\nabla': '∇',  # \nabla -> ∇
            r'\\forall': '∀',  # \forall -> ∀
            r'\\exists': '∃',  # \exists -> ∃
            r'\\in': '∈',  # \in -> ∈
            r'\\notin': '∉',  # \notin -> ∉
            r'\\subset': '⊂',  # \subset -> ⊂
            r'\\supset': '⊃',  # \supset -> ⊃
            r'\\cup': '∪',  # \cup -> ∪
            r'\\cap': '∩',  # \cap -> ∩
            r'\\emptyset': '∅',  # \emptyset -> ∅
            r'\\rightarrow': '→',  # \rightarrow -> →
            r'\\leftarrow': '←',  # \leftarrow -> ←
            r'\\leftrightarrow': '↔',  # \leftrightarrow -> ↔
            r'\\Rightarrow': '⇒',  # \Rightarrow -> ⇒
            r'\\Leftarrow': '⇐',  # \Leftarrow -> ⇐
            r'\\Leftrightarrow': '⇔',  # \Leftrightarrow -> ⇔
        }
        
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text)
        
        # Clean up extra spaces and formatting
        text = re.sub(r'\s+', ' ', text)  # Multiple spaces -> single space
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # **bold** -> bold
        text = re.sub(r'\*(.*?)\*', r'\1', text)  # *italic* -> italic
        
        # Clean up common OCR artifacts and brackets/braces
        text = text.replace('\\', '')  # Remove stray backslashes
        text = text.replace('{', '').replace('}', '')  # Remove braces
        text = text.replace('[', '').replace(']', '')  # Remove brackets
        
        # Fix common OCR issues with mathematical symbols
        text = text.replace('???', '')  # Remove question mark placeholders
        text = text.replace('??', '')   # Remove double question marks
        text = text.replace('?', '')    # Remove single question marks that might be OCR artifacts
        
        # Ensure proper spacing around mathematical operators
        text = re.sub(r'(\d+)\s*×\s*(\d+)', r'\1 × \2', text)  # Add spaces around ×
        text = re.sub(r'(\d+)\s*÷\s*(\d+)', r'\1 ÷ \2', text)  # Add spaces around ÷
        text = re.sub(r'(\d+)\s*\+\s*(\d+)', r'\1 + \2', text)  # Add spaces around +
        text = re.sub(r'(\d+)\s*-\s*(\d+)', r'\1 - \2', text)  # Add spaces around -
        
        # Additional mathematical symbol fixes
        text = text.replace('\\boxed{', '').replace('}', '')  # Remove \boxed{} wrapper
        text = text.replace('\\text{', '').replace('}', '')   # Remove \text{} wrapper
        text = text.replace('\\mathrm{', '').replace('}', '') # Remove \mathrm{} wrapper
        
        # Fix common LaTeX command issues
        text = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', text)  # Remove any remaining LaTeX commands with braces
        
        # Structure multi-step solutions with proper line breaks
        lines = text.split('\n')
        structured_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Remove step numbers (1., 2., 3., etc.) and clean up the line
            line = re.sub(r'^\d+\.\s*', '', line)
            
            # Skip lines that are just step indicators
            if line.lower().startswith('step') and len(line) < 20:
                continue
                
            # Add spacing before key mathematical operations
            if (line.startswith('Calculate') or line.startswith('Then') or 
                line.startswith('Next') or line.startswith('Finally') or
                line.startswith('Subtract') or line.startswith('Divide') or
                line.startswith('Add') or line.startswith('Multiply')):
                if structured_lines and structured_lines[-1].strip():
                    structured_lines.append('')
                structured_lines.append(line)
            elif '=' in line and any(op in line for op in ['+', '-', '×', '÷', '/']):
                # Mathematical equations - add spacing
                if structured_lines and structured_lines[-1].strip():
                    structured_lines.append('')
                structured_lines.append(line)
            elif line.lower().startswith('answer') or line.lower().startswith('solution') or line.lower().startswith('result'):
                # Final answer - add extra spacing and make it prominent
                if structured_lines and structured_lines[-1].strip():
                    structured_lines.append('')
                structured_lines.append('')
                structured_lines.append(line)
                structured_lines.append('')
            else:
                structured_lines.append(line)
        
        # Join with proper spacing
        result = '\n'.join(structured_lines)
        
        # Clean up multiple consecutive empty lines
        result = re.sub(r'\n\s*\n\s*\n', '\n\n', result)
        
        return result.strip()

    def send_image_to_llm(self, crop_bgr: np.ndarray) -> Optional[str]:
        """Send cropped image and OCR text to an LLM and return the response text."""
        try:
            # Encode as base64 data URL
            success, buf = cv2.imencode('.jpg', crop_bgr)
            if not success:
                print("Error: Failed to encode crop for LLM")
                return None
            b64 = base64.b64encode(buf.tobytes()).decode('utf-8')
            data_url = f"data:image/jpeg;base64,{b64}"
            recognized_text = self.ocr_extract_text(crop_bgr)

            # Prefer provider based on available keys / env
            provider = os.getenv('LLM_PROVIDER', '').strip().lower()
            openai_key = os.getenv('OPENAI_API_KEY')
            anthropic_key = os.getenv('ANTHROPIC_API_KEY')
            had_provider = False
            last_error_message = None

            if (provider == 'openai' or not provider) and openai_key:
                try:
                    from openai import OpenAI
                    org_id = os.getenv('OPENAI_ORG_ID')
                    client = OpenAI(api_key=openai_key, organization=org_id) if org_id else OpenAI(api_key=openai_key)
                    prompt = "Solve this problem:"
                    resp = client.chat.completions.create(
                        model=os.getenv('OPENAI_VISION_MODEL', 'gpt-4o-mini'),
                        messages=[{
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": data_url}},
                                *([{ "type": "text", "text": f"OCR (may be noisy): {recognized_text}"}] if recognized_text else [])
                            ]
                        }],
                        temperature=0.0,
                    )
                    text = resp.choices[0].message.content if resp.choices else None
                    if text:
                        return text
                except Exception as e:
                    last_error_message = f"OpenAI call failed: {e}"
                    print(last_error_message)
                finally:
                    had_provider = True

            if anthropic_key:
                try:
                    import anthropic
                    client = anthropic.Anthropic(api_key=anthropic_key)
                    prompt = "Solve this problem:"
                    msg = client.messages.create(
                        model=os.getenv('ANTHROPIC_VISION_MODEL', 'claude-3-5-sonnet-20240620'),
                        max_tokens=1024,
                        temperature=0.0,
                        messages=[{
                            "role": "user",
                            "content": [
                                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                                {"type": "text", "text": prompt},
                                *([{ "type": "text", "text": f"OCR (may be noisy): {recognized_text}"}] if recognized_text else [])
                            ]
                        }]
                    )
                    # Anthropic SDK returns list of content blocks. Extract text blocks
                    parts = []
                    for block in msg.content:
                        if getattr(block, 'type', None) == 'text':
                            parts.append(getattr(block, 'text', ''))
                    return "\n".join(p for p in parts if p)
                except Exception as e:
                    last_error_message = f"Anthropic call failed: {e}"
                    print(last_error_message)
                finally:
                    had_provider = True

            if not had_provider:
                print("No LLM provider configured. Set OPENAI_API_KEY or ANTHROPIC_API_KEY.")
            else:
                if last_error_message:
                    print(f"LLM call did not succeed: {last_error_message}")
            return None
        except Exception as e:
            print(f"Error preparing image for LLM: {e}")
            return None

    def send_multiple_images_to_llm(self, parts: List[dict]) -> Optional[str]:
        """Send multiple cropped images (and optional OCR texts) to the LLM in one request."""
        try:
            # Prepare OpenAI-style content array
            content_blocks = []
            content_blocks.append({"type": "text", "text": "Solve this question and output the answer:"})
            for i, p in enumerate(parts, start=1):
                img = p.get("img")
                text_hint = p.get("text", "")
                success, buf = cv2.imencode('.jpg', img)
                if not success:
                    continue
                b64 = base64.b64encode(buf.tobytes()).decode('utf-8')
                data_url = f"data:image/jpeg;base64,{b64}"
                content_blocks.append({"type": "image_url", "image_url": {"url": data_url}})
                if text_hint:
                    content_blocks.append({"type": "text", "text": f"OCR (may be noisy) part {i}: {text_hint}"})

            provider = os.getenv('LLM_PROVIDER', '').strip().lower()
            openai_key = os.getenv('OPENAI_API_KEY')
            anthropic_key = os.getenv('ANTHROPIC_API_KEY')
            had_provider = False
            last_error_message = None

            if (provider == 'openai' or not provider) and openai_key:
                try:
                    from openai import OpenAI
                    org_id = os.getenv('OPENAI_ORG_ID')
                    client = OpenAI(api_key=openai_key, organization=org_id) if org_id else OpenAI(api_key=openai_key)
                    resp = client.chat.completions.create(
                        model=os.getenv('OPENAI_VISION_MODEL', 'gpt-4o-mini'),
                        messages=[{"role": "user", "content": content_blocks}],
                        temperature=0.0,
                    )
                    text = resp.choices[0].message.content if resp.choices else None
                    if text:
                        return text
                except Exception as e:
                    last_error_message = f"OpenAI call failed: {e}"
                    print(last_error_message)
                finally:
                    had_provider = True

            if anthropic_key:
                try:
                    import anthropic
                    client = anthropic.Anthropic(api_key=anthropic_key)
                    # Anthropic expects a different content format
                    anthro_content = []
                    for block in content_blocks:
                        if block.get('type') == 'text':
                            anthro_content.append({"type": "text", "text": block.get('text', '')})
                        elif block.get('type') == 'image_url':
                            url = block.get('image_url', {}).get('url', '')
                            if url.startswith('data:image/jpeg;base64,'):
                                b64 = url.split(',', 1)[1]
                                anthro_content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}})
                    msg = client.messages.create(
                        model=os.getenv('ANTHROPIC_VISION_MODEL', 'claude-3-5-sonnet-20240620'),
                        max_tokens=1024,
                        temperature=0.0,
                        messages=[{"role": "user", "content": anthro_content}],
                    )
                    parts_out = []
                    for block in msg.content:
                        if getattr(block, 'type', None) == 'text':
                            parts_out.append(getattr(block, 'text', ''))
                    return "\n".join(p for p in parts_out if p)
                except Exception as e:
                    last_error_message = f"Anthropic call failed: {e}"
                    print(last_error_message)
                finally:
                    had_provider = True

            if not had_provider:
                print("No LLM provider configured. Set OPENAI_API_KEY or ANTHROPIC_API_KEY.")
            else:
                if last_error_message:
                    print(f"LLM call did not succeed: {last_error_message}")
            return None
        except Exception as e:
            print(f"Error preparing multi-image request for LLM: {e}")
            return None
    
    def capture_image_with_frame(self, processed_frame):
        """Capture a processed image with text detection boxes"""
        if self.camera is None:
            print("Error: Camera not initialized")
            return None
        
        try:
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"text_detection_{timestamp}.jpg"
            filepath = os.path.join(self.output_dir, filename)
            
            # Save the processed image with text detection boxes
            success = cv2.imwrite(filepath, processed_frame)
            if not success:
                print("Error: Could not save image")
                return None
            
            self.captured_images.append(filepath)
            print(f"✓ Image captured and saved: {os.path.basename(filepath)}")
            return filepath, processed_frame
            
        except Exception as e:
            print(f"Error capturing image: {e}")
            return None
    
    def capture_image(self):
        """Capture a single image from the camera with error handling"""
        if self.camera is None:
            print("Error: Camera not initialized")
            return None
        
        try:
            ret, frame = self.camera.read()
            if not ret:
                print("Error: Could not capture frame")
                return None
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"text_detection_{timestamp}.jpg"
            filepath = os.path.join(self.output_dir, filename)
            
            # Save the image with error handling
            success = cv2.imwrite(filepath, frame)
            if not success:
                print("Error: Could not save image")
                return None
            
            self.captured_images.append(filepath)
            print(f"✓ Image captured and saved: {os.path.basename(filepath)}")
            return filepath, frame
            
        except Exception as e:
            print(f"Error capturing image: {e}")
            return None
    
    def run_camera_feed(self):
        """Run the real-time camera feed with text detection"""
        if not self.start_camera():
            return
        
        print("\n=== Text Detection Camera Controls ===")
        print("• Press 't' to toggle text detection on/off")
        print("• Press 'a' to toggle accurate mode (slower but more accurate)")
        print("• Press 'g' to cycle granularity (paragraph/line/word)")
        print("• Press 'f' to show/save full recognized text (Vision)")
        print("• Press 'q' to quit")
        print("• Close the camera window to quit")
        print("=====================================\n")
        
        # Show detection strategy (original style)
        if self.accurate_mode:
            if ('easyocr' in sys.modules) and isinstance(self.ocr_reader, easyocr.Reader):
                print("✓ Accurate mode: EasyOCR (reads text)")
            elif self.ocr_reader == 'tesseract':
                print("✓ Accurate mode: Tesseract (reads text)")
            else:
                print("⚠ Accurate mode requested but OCR not available; using fast contour detection")
        else:
            print("✓ Fast mode: contour-based detection (no OCR) for higher FPS")
        
        print()
        
        # Create named window for better control
        cv2.namedWindow('Text Detection Camera', cv2.WINDOW_NORMAL)
        
        frame_count = 0
        last_capture_time = 0
        text_count = 0
        last_detection_time = time.time()
        detection_results = None  # Store last detection results (not used with async)
        detection_frame_count = 0  # Track when to run detection (not used with async)

        # Start background detection
        self._start_detection_thread()
        
        while self.running:
            try:
                ret, frame = self.camera.read()
                if not ret:
                    print("Error: Could not read frame from camera")
                    break
                
                # Create a copy of the frame for display
                display_frame = frame
                
                # Offer frames to background detector, and draw last known boxes every frame
                if self.detection_enabled and self.ocr_reader is not None:
                    self._offer_frame_to_detector(frame.copy())
                    # Draw last detected regions onto the display frame
                    if self.last_text_regions:
                        self._draw_regions(display_frame, self.last_text_regions)
                        text_count = len(self.last_text_regions)
                    else:
                        text_count = 0
                else:
                    text_count = 0
                    # Clear cached regions when detection is disabled
                    if self.last_text_regions:
                        self.last_text_regions = []
                
                # Get frame dimensions for text positioning
                h, w = display_frame.shape[:2]
                
                # Add instructions to the frame (moved to right side)
                cv2.putText(display_frame, "Press 't' to toggle, 'a' for accurate mode", 
                           (w - 350, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(display_frame, f"Captured: {len(self.captured_images)}", 
                           (w - 200, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Add detection status
                status = "ON" if self.detection_enabled else "OFF"
                cv2.putText(display_frame, f"Detection: {status}", 
                           (w - 200, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # Add mode status
                mode = "ACCURATE" if self.accurate_mode else "FAST"
                cv2.putText(display_frame, f"Mode: {mode}", 
                           (w - 200, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # Add text count with visibility status
                if text_count > 0:
                    cv2.putText(display_frame, f"Text regions: {text_count}", 
                               (w - 200, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                else:
                    cv2.putText(display_frame, "No text detected", 
                               (w - 200, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)
                
                # Center detection area indicator removed for split-screen layout

                # Autosolve timer logic: require stable regions in center area for N seconds
                try:
                    frame_h, frame_w = frame.shape[:2]
                    
                    # Check if text is currently visible on screen (real-time detection)
                    # Only check every few frames to reduce sensitivity to minor movements
                    if frame_count % 10 == 0:  # Check every 10 frames (about 0.3 seconds at 30fps)
                        current_signature = self._compute_regions_signature(self.last_text_regions, frame_w, frame_h)
                    else:
                        # Use the last computed signature to avoid constant changes
                        current_signature = getattr(self, '_last_computed_signature', "")
                    
                    # Store the current signature for next frame
                    self._last_computed_signature = current_signature
                    now = time.time()
                    
                    # Check if we're in cooldown period after a solution
                    time_since_last_solution = now - self.last_llm_solution_time
                    in_cooldown = time_since_last_solution < self.solution_cooldown_seconds
                    
                    # If no text is currently visible, clear cached regions and reset
                    if not current_signature:
                        # Clear cached regions when text is no longer visible
                        if self.last_text_regions:
                            self.last_text_regions = []
                            print("Text no longer visible - cleared cached regions")
                        
                        # Reset autosolve state when text disappears
                        self._hold_signature = None
                        self._hold_start_time = 0.0
                        self._pending_signature = None
                        self._pending_since = 0.0
                    
                    # Only proceed if text is currently visible and not in cooldown
                    if current_signature and not in_cooldown:
                        if self._hold_signature is None:
                            # Start countdown immediately on first non-empty detection
                            self._hold_signature = current_signature
                            self._hold_start_time = now
                            self._pending_signature = None
                            self._pending_since = 0.0
                        else:
                            # Check if the current signature is significantly different from the held one
                            # Use a much more tolerant comparison to handle camera shake
                            if current_signature != self._hold_signature:
                                # Only reset if the difference is significant (not just camera shake)
                                # Use a longer stability period and more tolerant comparison
                                if self._pending_signature is None:
                                    # First time seeing this signature - start pending timer
                                    self._pending_signature = current_signature
                                    self._pending_since = now
                                elif current_signature == self._pending_signature:
                                    # Same pending signature for a while - check if it's stable
                                    if now - self._pending_since > 2.5:  # 2.5 second stability check
                                        # Signature has been stable for 2.5 seconds - accept it as new text
                                        self._hold_signature = current_signature
                                        self._hold_start_time = now
                                        self._pending_signature = None
                                        self._pending_since = 0.0
                                        print("Text changed - resetting timer")
                                else:
                                    # Different pending signature - update pending
                                    self._pending_signature = current_signature
                                    self._pending_since = now

                        hold_secs = now - self._hold_start_time
                        # Show "Solution Processing" instead of countdown
                        if (not self._autosolve_in_progress) and hold_secs >= self.autosolve_hold_seconds:
                            # Clear previous solution when starting new solve
                            self.last_llm_solution_text = None
                            # Snapshot current frame to crop parts and start autosolve
                            self._start_autosolve(self.last_text_regions, frame.copy())
                            # Show processing message
                            cv2.putText(display_frame, "Capturing complete equation...", (w - 250, 180),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 215, 255), 2)
                        elif self._autosolve_in_progress:
                            # Show processing message while solving
                            cv2.putText(display_frame, "Solution Processing...", (w - 200, 180),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 215, 255), 2)
                        else:
                            # Show countdown message
                            remaining_time = max(0.0, self.autosolve_hold_seconds - hold_secs)
                            cv2.putText(display_frame, f"Hold steady: {remaining_time:.1f}s", (w - 200, 180),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 215, 255), 2)
                    elif in_cooldown:
                        # Show cooldown message
                        remaining_cooldown = max(0.0, self.solution_cooldown_seconds - time_since_last_solution)
                        cv2.putText(display_frame, f"Cooldown: {remaining_cooldown:0.1f}s", (w - 200, 180),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
                        # Reset autosolve state during cooldown
                        self._hold_signature = None
                        self._hold_start_time = 0.0
                        self._pending_signature = None
                        self._pending_since = 0.0
                    else:
                        # No signature in center; reset
                        self._hold_signature = None
                        self._hold_start_time = 0.0
                        self._pending_signature = None
                        self._pending_since = 0.0
                except Exception as e:
                    print(f"Error in autosolve logic: {e}")
                    pass
                
                # Display the frame
                # Split screen: OCR reading on left (3/5), solution on right (2/5)
                left_w = int(w * 0.6)  # 3/5 of screen width
                mid_x = left_w + 10
                
                # Draw vertical divider
                cv2.line(display_frame, (mid_x, 0), (mid_x, h), (100, 100, 100), 2)
                
                # Left side: Show OCR reading area (3/5 of screen)
                left_h = h - 80  # Reduced height
                left_x = 10
                left_y = 40  # Start below top
                
                # Draw OCR reading area with reduced borders
                cv2.rectangle(display_frame, (left_x, left_y), (left_x + left_w, left_y + left_h), (0, 255, 255), 2)
                
                # Right side: Show solution if available, or processing message
                if self._autosolve_in_progress and not self.last_llm_solution_text:
                    # Show "Solution Processing" in the solution area
                    try:
                        # Position processing box on right side (centered)
                        right_w = w - mid_x - 20
                        box_w = right_w - 20
                        box_h = 120  # Fixed height for processing message
                        x0 = mid_x + 10
                        y0 = (h - box_h) // 2  # Center vertically
                        x1 = x0 + box_w
                        y1 = y0 + box_h
                        
                        # Draw processing box with transparent background
                        overlay = display_frame.copy()
                        cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 0), -1)
                        cv2.addWeighted(overlay, 0.3, display_frame, 0.7, 0, display_frame)
                        
                        # Add purple border
                        cv2.rectangle(display_frame, (x0, y0), (x1, y1), (255, 0, 255), 2)
                        
                        # Processing message (centered)
                        processing_text = "Solution Processing..."
                        text_size = cv2.getTextSize(processing_text, cv2.FONT_HERSHEY_DUPLEX, 0.8, 2)[0]
                        text_width = text_size[0]
                        text_x = x0 + (box_w - text_width) // 2
                        text_y = y0 + (box_h + text_size[1]) // 2
                        
                        cv2.putText(display_frame, processing_text, (text_x, text_y), 
                                   cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
                    except Exception:
                        pass
                elif self.last_llm_solution_text:
                    try:
                        # Clean up the solution text (remove LaTeX formatting)
                        clean_text = self._clean_solution_text(self.last_llm_solution_text)
                        
                        # Wrap text for right side (2/5 of screen)
                        right_w = w - mid_x - 20
                        max_chars = max(25, int(right_w / 16))
                        lines = []
                        for paragraph in clean_text.splitlines():
                            p = paragraph.strip()
                            if not p:
                                lines.append("")
                                continue
                            while len(p) > max_chars:
                                # Break at last space within limit
                                brk = p.rfind(' ', 0, max_chars)
                                if brk <= 0:
                                    brk = max_chars
                                lines.append(p[:brk].strip())
                                p = p[brk:].strip()
                            if p:
                                lines.append(p)
                        
                        # Position solution box on right side (centered)
                        line_h = 35  # Increased line height for better spacing
                        box_w = right_w - 20
                        box_h = min(h - 40, len(lines) * line_h + 80)  # More padding
                        x0 = mid_x + 10
                        y0 = (h - box_h) // 2  # Center vertically
                        x1 = x0 + box_w
                        y1 = y0 + box_h
                        
                        # Draw solution with transparent background
                        # Create a semi-transparent overlay
                        overlay = display_frame.copy()
                        cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 0), -1)
                        cv2.addWeighted(overlay, 0.3, display_frame, 0.7, 0, display_frame)
                        
                        # Add purple border
                        cv2.rectangle(display_frame, (x0, y0), (x1, y1), (255, 0, 255), 2)
                        
                        # Header (centered)
                        header_y = y0 + 35
                        header_text = "AI SOLUTION"
                        header_size = cv2.getTextSize(header_text, cv2.FONT_HERSHEY_DUPLEX, 0.8, 2)[0]
                        header_width = header_size[0]
                        header_x = x0 + (box_w - header_width) // 2
                        cv2.putText(display_frame, header_text, (header_x, header_y), 
                                   cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 0), 2)
                        cv2.putText(display_frame, header_text, (header_x, header_y), 
                                   cv2.FONT_HERSHEY_DUPLEX, 0.8, (100, 200, 255), 1)
                        
                        # Body text (centered)
                        y_txt = y0 + 75
                        for i, ln in enumerate(lines):
                            # Use consistent white text for better readability on transparent background
                            color = (255, 255, 255)  # White text
                            
                            # Center each line of text horizontally
                            text_size = cv2.getTextSize(ln, cv2.FONT_HERSHEY_DUPLEX, 0.7, 1)[0]
                            text_width = text_size[0]
                            text_x = x0 + (box_w - text_width) // 2
                            
                            # Add subtle shadow for depth
                            cv2.putText(display_frame, ln, (text_x + 1, y_txt + 1), 
                                       cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 2)
                            cv2.putText(display_frame, ln, (text_x, y_txt), 
                                       cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 1)
                            y_txt += line_h
                    except Exception:
                        pass

                cv2.imshow('Text Detection Camera', display_frame)
                
                # Handle key presses and window close
                key = cv2.waitKey(1) & 0xFF
                
                # Check if window was closed
                try:
                    if cv2.getWindowProperty('Text Detection Camera', cv2.WND_PROP_VISIBLE) < 1:
                        print("Camera window closed. Quitting...")
                        break
                except:
                    # Window might not exist anymore
                    break
                
                if key == ord('q'):
                    print("Quitting camera feed...")
                    break
                elif key == ord('t'):
                    self.detection_enabled = not self.detection_enabled
                    status = "enabled" if self.detection_enabled else "disabled"
                    print(f"Text detection {status}")
                elif key == ord('a'):
                    self.accurate_mode = not self.accurate_mode
                    mode = "accurate" if self.accurate_mode else "fast"
                    print(f"Switched to {mode} mode")
                elif key == ord('g'):
                    self.granularity = (
                        'line' if self.granularity == 'paragraph' else
                        'word' if self.granularity == 'line' else
                        'paragraph'
                    )
                    print(f"Granularity set to: {self.granularity}")
                elif key == ord('f'):
                    # Show and save last full text recognized by Vision
                    if self.last_full_text:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        txt_path = os.path.join(self.output_dir, f"full_text_{timestamp}.txt")
                        try:
                            with open(txt_path, 'w') as f:
                                f.write(self.last_full_text)
                            print(f"✓ Saved full text to {os.path.basename(txt_path)}")
                        except Exception as e:
                            print(f"Could not save full text: {e}")
                        print("\n=== Recognized Text (Vision) ===\n" + self.last_full_text + "\n===============================\n")
                    else:
                        print("No full text available yet. Ensure Vision OCR is active and in view.")


                
                frame_count += 1
                
            except Exception as e:
                print(f"Error in camera loop: {e}")
                break
        
        self.cleanup()
    
    def cleanup(self):
        """Clean up camera resources with error handling"""
        try:
            self._stop_detection_thread()
            if self.camera is not None:
                self.camera.release()
                self.camera = None
        except Exception as e:
            print(f"Error releasing camera: {e}")
        
        try:
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Error destroying windows: {e}")
    
    def get_captured_images(self):
        """Return list of captured image paths"""
        return self.captured_images

def main():
    """Main function to run the text detection camera application"""
    print("=== Real-time Text Detection Camera ===")
    print("This application detects text in real-time and draws blue boxes around it.")
    print("Perfect for capturing math problems and other text content.")
    print()
    
    if not OCR_AVAILABLE:
        print("⚠ WARNING: No OCR library found!")
        print("To enable text detection, install one of the following:")
        print("  pip install easyocr")
        print("  pip install pytesseract")
        print("The camera will still work, but without text detection.")
        print()
    
    app = TextDetectionCamera()
    
    try:
        app.run_camera_feed()
        
        # After quitting, show captured images
        if app.captured_images:
            print(f"\nTotal images captured: {len(app.captured_images)}")
            print("Images saved in:", app.output_dir)
        
        print("\nApplication completed successfully!")
        
    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
        app.cleanup()
    except Exception as e:
        print(f"Unexpected error: {e}")
        app.cleanup()
    finally:
        # Ensure cleanup happens
        app.cleanup()

if __name__ == "__main__":
    main() 