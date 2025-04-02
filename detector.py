import os
import time
import logging
import cv2
import numpy as np
import random

class DeepfakeDetector:
    def __init__(self):
        """
        Initialize the deepfake detector.
        For this simplified demo, we're using a basic approach without a deep learning model.
        """
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        logging.info("Deepfake detector initialized")

    def extract_faces(self, frame):
        """
        Extract faces from a frame using a face detector.
        Returns a list of face images.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        face_images = []
        for (x, y, w, h) in faces:
            # Make the bounding box slightly larger to include the whole face
            x, y, w, h = max(0, x-10), max(0, y-10), w+20, h+20
            face = frame[y:y+h, x:x+w]
            if face.size > 0:  # Ensure the face image is not empty
                face_images.append(face)
                
        return face_images

    def analyze_face_for_artifacts(self, face):
        """
        Simplified analysis of a face for potential deepfake artifacts.
        In a real implementation, this would use ML-based detection.
        
        For this demo, we'll use simple image processing metrics as a stand-in.
        """
        # Convert to grayscale for simpler analysis
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        
        # Calculate basic image statistics
        mean_val = np.mean(gray)
        std_dev = np.std(gray)
        
        # Apply edge detection to find facial feature edges
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        # Calculate texture uniformity (simple texture analysis)
        glcm = self.calculate_glcm(gray)
        uniformity = np.sum(glcm**2)
        
        # Create a heuristic score based on these features
        # This would normally be calculated by a trained classifier
        # We're using a weighted combination for demonstration
        score = 0.2 * (mean_val / 255.0) + 0.3 * (1.0 - std_dev / 128.0) + 0.25 * edge_density + 0.25 * uniformity
        
        # Add some randomness to simulate variation in detection
        variation = random.uniform(-0.15, 0.15)
        score = max(0.0, min(1.0, score + variation))
        
        return score
    
    def calculate_glcm(self, gray_img):
        """
        Calculate a simple Gray Level Co-occurrence Matrix for texture analysis.
        Simplified version for demonstration.
        """
        # Downsample for faster processing
        small = cv2.resize(gray_img, (32, 32))
        # Quantize to fewer gray levels
        levels = 8
        small = np.uint8(np.floor(small / (256 / levels)) * (256 / levels))
        
        # Simple GLCM calculation (horizontal adjacency only)
        glcm = np.zeros((levels, levels))
        for i in range(small.shape[0]):
            for j in range(small.shape[1]-1):
                i_level = int(small[i, j] * levels / 256)
                j_level = int(small[i, j+1] * levels / 256)
                glcm[i_level, j_level] += 1
        
        # Normalize
        if np.sum(glcm) > 0:
            glcm = glcm / np.sum(glcm)
        
        return glcm

    def predict_frame(self, frame):
        """
        Predict whether a frame contains deepfake content.
        Returns a probability between 0 (real) and 1 (fake).
        """
        # Extract faces from the frame
        faces = self.extract_faces(frame)
        
        if not faces:
            return 0.5  # If no faces found, return uncertain prediction
        
        # Process each face and get predictions
        predictions = []
        for face in faces:
            prediction = self.analyze_face_for_artifacts(face)
            predictions.append(prediction)
        
        # Return the average prediction across all faces
        return np.mean(predictions) if predictions else 0.5

    def analyze_video(self, video_path):
        """
        Analyze a video file for deepfake content.
        Returns a dictionary with the analysis results.
        """
        start_time = time.time()
        video = None
        
        try:
            # Verify the file exists
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")
                
            # Check file size
            file_size = os.path.getsize(video_path)
            if file_size == 0:
                raise ValueError(f"Video file is empty: {video_path}")
            
            logging.info(f"Starting analysis of video: {video_path}, size: {file_size} bytes")
            
            # Open the video file
            video = cv2.VideoCapture(video_path)
            if not video.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")
            
            total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = video.get(cv2.CAP_PROP_FPS)
            
            logging.info(f"Video properties: {total_frames} frames, {fps} FPS")
            
            # Validate video properties
            if total_frames <= 0 or fps <= 0:
                logging.warning(f"Suspicious video properties: frames={total_frames}, fps={fps}")
                # Continue anyway, but with default values
                if total_frames <= 0:
                    total_frames = 30  # Assume at least one second
                if fps <= 0:
                    fps = 30  # Assume standard framerate
            
            # Calculate frame sampling rate - analyze approximately one frame per second
            sampling_rate = max(1, int(fps))
            
            frame_count = 0
            analyzed_count = 0
            predictions = []
            
            # Process video frames
            while True:
                ret, frame = video.read()
                if not ret:
                    break
                
                # Sample frames at the specified rate
                if frame_count % sampling_rate == 0:
                    try:
                        prediction = self.predict_frame(frame)
                        predictions.append(prediction)
                        analyzed_count += 1
                        
                        # Log progress
                        if analyzed_count % 5 == 0:
                            logging.debug(f"Analyzed {analyzed_count} frames so far")
                        
                        # Limit the number of analyzed frames for demo purposes
                        if analyzed_count >= 10:
                            break
                    except Exception as frame_error:
                        logging.warning(f"Error analyzing frame {frame_count}: {str(frame_error)}")
                        # Continue to next frame
                
                frame_count += 1
                
                # For demo purposes, limit total frames processed
                if frame_count >= 100:
                    break
            
            # Calculate average prediction
            avg_prediction = np.mean(predictions) if predictions else 0.5
            
            # Determine if the video is fake based on the prediction
            # Threshold can be adjusted based on model performance
            is_fake = avg_prediction > 0.5
            
            # Calculate probabilities
            fake_probability = float(avg_prediction)
            real_probability = float(1.0 - avg_prediction)
            
            # Calculate confidence - how far from 0.5 (uncertain) the prediction is
            confidence = abs(avg_prediction - 0.5) * 2  # Scale to 0-1
            
            # Format the output
            processing_time = time.time() - start_time
            
            result = {
                'real_probability': real_probability,
                'fake_probability': fake_probability,
                'is_fake': is_fake,
                'confidence': confidence,
                'processing_time': processing_time,
                'analyzed_frames': analyzed_count
            }
            
            logging.info(f"Analysis complete: is_fake={is_fake}, confidence={confidence:.2f}, time={processing_time:.2f}s")
            return result
            
        except Exception as e:
            logging.error(f"Error during video analysis: {str(e)}", exc_info=True)
            raise
            
        finally:
            # Ensure video is released even if an exception occurs
            if video is not None:
                video.release()
