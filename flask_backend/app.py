import os
import cv2
import json
import numpy as np
import tempfile
import shutil
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.exceptions import RequestEntityTooLarge
import dlib
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import preprocess_input

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB limit
CORS(app, origins=["http://localhost:3000"], supports_credentials=True)  # Enable CORS for React frontend

# Constants
IMG_SIZE = (299, 299)
MOTION_THRESHOLD = 20
FRAME_SKIP = 2
no_of_frames = 10
MAX_FRAMES = no_of_frames

# Initialize Dlib's frontal face detector
detector = dlib.get_frontal_face_detector()

# Load the trained model
def load_deepfake_model():
    """Load the trained deepfake detection model"""
    try:
        # Try loading from different model files
        model_paths = [
            '/Users/nhz/Desktop/deepfake/trained_model/deepfake_detection_model.h5',
            '/Users/nhz/Desktop/deepfake/trained_model/deepfake_detection_model.keras',
            '/Users/nhz/Desktop/deepfake/deepfake-detection/InceptionV3_LSTM_GRU/trained_model/deepfake_detection_model.h5'
        ]
        
        model = None
        for model_path in model_paths:
            if not os.path.exists(model_path):
                print(f"⚠️  Model file not found: {model_path}")
                continue
                
            print(f"🎯 Trying model: {model_path}")
            
            try:
                # Method 1: Load with architecture reconstruction
                print("🔄 Method 1: Loading with architecture reconstruction...")
                
                # Load architecture from JSON if available
                json_path = model_path.replace('.h5', '.json').replace('.keras', '.json')
                json_path = json_path.replace('deepfake_detection_model', 'model_architecture')
                
                if os.path.exists(json_path):
                    print(f"📋 Loading architecture from: {json_path}")
                    with open(json_path, 'r') as json_file:
                        model_json = json_file.read()
                    
                    # Create model from JSON with explicit input shape
                    from tensorflow.keras.models import model_from_json
                    from tensorflow.keras.layers import TimeDistributed
                    
                    custom_objects = {
                        'TimeDistributed': TimeDistributed
                    }
                    
                    model = model_from_json(model_json, custom_objects=custom_objects)
                    
                    # Load weights
                    weights_path = model_path.replace('.keras', '.h5')  # Ensure we use .h5 for weights
                    if os.path.exists(weights_path):
                        model.load_weights(weights_path)
                        print("✅ Weights loaded successfully!")
                    else:
                        print(f"⚠️  Weights file not found: {weights_path}")
                        continue
                        
                    # Compile the model
                    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
                    break
                    
            except Exception as e1:
                print(f"⚠️  Architecture reconstruction failed: {e1}")
                
                # Method 2: Direct loading with custom objects
                try:
                    print("🔄 Method 2: Direct loading with custom objects...")
                    custom_objects = {
                        'TimeDistributed': tf.keras.layers.TimeDistributed
                    }
                    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
                    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
                    break
                    
                except Exception as e2:
                    print(f"⚠️  Direct loading failed: {e2}")
                    
                    # Method 3: Rebuild model manually
                    try:
                        print("🔄 Method 3: Manual model reconstruction...")
                        model = rebuild_model_manually()
                        if model is not None:
                            # Try to load weights
                            weights_path = model_path
                            model.load_weights(weights_path)
                            print("✅ Manual reconstruction successful!")
                            break
                    except Exception as e3:
                        print(f"⚠️  Manual reconstruction failed: {e3}")
                        continue
        
        if model is None:
            print("❌ All model loading methods failed!")
            return None
        
        print("✅ Model loaded successfully!")
        print(f"📊 Model input shape: {model.input_shape}")
        print(f"📊 Model output shape: {model.output_shape}")
        
        # Test the model with a dummy input to ensure it works
        try:
            dummy_input = tf.random.normal((1, 10, 299, 299, 3))
            _ = model(dummy_input, training=False)
            print("✅ Model test prediction successful!")
        except Exception as test_error:
            print(f"⚠️  Model test failed: {test_error}")
            # Try with different input shape
            try:
                dummy_input = tf.random.normal((1, no_of_frames, 299, 299, 3))
                _ = model(dummy_input, training=False)
                print("✅ Model test with fixed frames successful!")
            except Exception as test_error2:
                print(f"⚠️  Model test with fixed frames failed: {test_error2}")
                print("Model loaded but may have issues with prediction")
        
        return model
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def rebuild_model_manually():
    """Manually rebuild the model architecture"""
    try:
        from tensorflow.keras.applications import InceptionV3
        from tensorflow.keras.layers import Dense, LSTM, GRU, TimeDistributed, Dropout, GlobalAveragePooling2D, Input
        from tensorflow.keras.models import Sequential
        
        print("🔨 Rebuilding model manually...")
        
        # Create the base InceptionV3 model
        base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
        
        # Freeze the base model
        base_model.trainable = False
        
        # Add global average pooling
        feature_extractor = Sequential([
            base_model,
            GlobalAveragePooling2D()
        ])
        
        # Build the sequential model with TimeDistributed
        model = Sequential()
        model.add(Input(shape=(no_of_frames, 299, 299, 3)))  # Fixed shape instead of None
        model.add(TimeDistributed(feature_extractor))
        model.add(LSTM(256, return_sequences=True))
        model.add(Dropout(0.5))
        model.add(LSTM(256))
        model.add(Dropout(0.5))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))
        
        # Compile the model
        model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
        
        print("✅ Manual model reconstruction complete!")
        return model
        
    except Exception as e:
        print(f"❌ Manual model reconstruction failed: {e}")
        return None

# Initialize model
model = load_deepfake_model()

def extract_faces_from_frame(frame, detector):
    """
    Detects faces in a frame and returns the resized faces.

    Parameters:
    - frame: The video frame to process.
    - detector: Dlib face detector.

    Returns:
    - resized_faces (list): List of resized faces detected in the frame.
    """
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray_frame)
    resized_faces = []

    for face in faces:
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        crop_img = frame[y1:y2, x1:x2]
        if crop_img.size != 0:  
            resized_face = cv2.resize(crop_img, IMG_SIZE)
            resized_faces.append(resized_face)

    # Debug: Log the number of faces detected
    #print(f"Detected {len(resized_faces)} faces in current frame")
    return resized_faces

def process_frame(video_path, detector, frame_skip):
    """
    Processes frames to extract motion and face data concurrently.

    Parameters:
    - video_path: Path to the video file.
    - detector: Dlib face detector.
    - frame_skip (int): Number of frames to skip for processing.

    Returns:
    - motion_frames (list): List of motion-based face images.
    - all_faces (list): List of all detected faces for fallback.
    """
    prev_frame = None
    frame_count = 0
    motion_frames = []
    all_faces = []
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames to improve processing speed
        if frame_count % frame_skip != 0:
            frame_count += 1
            continue

        # Debug: Log frame number being processed
        #print(f"Processing frame {frame_count}")

        # # Resize frame to reduce processing time (optional, adjust size as needed)
        # frame = cv2.resize(frame, (640, 360))

        # Extract faces from the current frame
        faces = extract_faces_from_frame(frame, detector)
        all_faces.extend(faces)  # Store all faces detected, including non-motion

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_frame is None:
            prev_frame = gray_frame
            frame_count += 1
            continue

        # Calculate frame difference to detect motion
        frame_diff = cv2.absdiff(prev_frame, gray_frame)
        motion_score = np.sum(frame_diff)

        # Debug: Log the motion score
        #print(f"Motion score: {motion_score}")

        # Check if motion is above the defined threshold and add the face to motion frames
        if motion_score > MOTION_THRESHOLD and faces:
            motion_frames.extend(faces)

        prev_frame = gray_frame
        frame_count += 1

    cap.release()
    return motion_frames, all_faces

def select_well_distributed_frames(motion_frames, all_faces, no_of_frames):
    """
    Selects well-distributed frames from the detected motion and fallback faces.

    Parameters:
    - motion_frames (list): List of frames with detected motion.
    - all_faces (list): List of all detected faces.
    - no_of_frames (int): Required number of frames.

    Returns:
    - final_frames (list): List of selected frames.
    """
    # Case 1: Motion frames exceed the required number
    if len(motion_frames) >= no_of_frames:
        interval = len(motion_frames) // no_of_frames
        distributed_motion_frames = [motion_frames[i * interval] for i in range(no_of_frames)]
        return distributed_motion_frames

    # Case 2: Motion frames are less than the required number
    needed_frames = no_of_frames - len(motion_frames)

    # If all frames together are still less than needed, return all frames available
    if len(motion_frames) + len(all_faces) < no_of_frames:
        #print(f"Returning all available frames: {len(motion_frames) + len(all_faces)}")
        return motion_frames + all_faces

    interval = max(1, len(all_faces) // needed_frames)
    additional_faces = [all_faces[i * interval] for i in range(needed_frames)]

    combined_frames = motion_frames + additional_faces
    interval = max(1, len(combined_frames) // no_of_frames)
    final_frames = [combined_frames[i * interval] for i in range(no_of_frames)]
    return final_frames

def extract_frames(no_of_frames, video_path):
    """Extract frames from video"""
    motion_frames, all_faces = process_frame(video_path, detector, FRAME_SKIP)
    final_frames = select_well_distributed_frames(motion_frames, all_faces, no_of_frames)
    return final_frames

def frames_to_base64(frames):
    """Convert frames to base64 strings for frontend display"""
    base64_frames = []
    for frame in frames:
        # Convert frame to uint8 if it's not already
        if frame.dtype != np.uint8:
            # Denormalize if the frame was preprocessed
            frame_display = ((frame + 1) * 127.5).astype(np.uint8) if frame.min() < 0 else frame.astype(np.uint8)
        else:
            frame_display = frame
        
        # Ensure frame is in BGR format for OpenCV
        if len(frame_display.shape) == 3 and frame_display.shape[2] == 3:
            # Convert RGB to BGR for OpenCV
            frame_display = cv2.cvtColor(frame_display, cv2.COLOR_RGB2BGR)
        
        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame_display, [cv2.IMWRITE_JPEG_QUALITY, 85])
        
        # Convert to base64
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        base64_frames.append(f"data:image/jpeg;base64,{frame_base64}")
    
    return base64_frames

def predict_video(model, video_path):
    """
    Predict if a video is REAL or FAKE using the trained model.

    Parameters:
    - model: The loaded deepfake detection model.
    - video_path: Path to the video file to be processed.

    Returns:
    - predicted_label: 'REAL' or 'FAKE' based on the model's prediction.
    - confidence: Confidence score for the prediction.
    - original_frames: List of extracted frames for display.
    - frames_base64: Base64 encoded frames for frontend display.
    """
    if model is None:
        return "UNKNOWN", 0.0, [], []
    
    # Extract frames from the video
    frames = extract_frames(no_of_frames, video_path)
    original_frames = frames.copy()

    # Convert original frames to base64 for frontend display
    frames_base64 = frames_to_base64(original_frames) if original_frames else []

    # Convert the frames list to a 5D tensor (1, time_steps, height, width, channels)
    if len(frames) < MAX_FRAMES:
        # Pad with zero arrays to match MAX_FRAMES
        while len(frames) < MAX_FRAMES:
            frames.append(np.zeros((299, 299, 3), dtype=np.float32))

    frames = frames[:MAX_FRAMES]
    frames = np.array(frames)    
    frames = preprocess_input(frames) 

    # Expand dims to fit the model input shape
    input_data = np.expand_dims(frames, axis=0)  # Shape becomes (1, MAX_FRAMES, 299, 299, 3)

    # Predict using the model
    prediction = model.predict(input_data)
    probability = prediction[0][0]  # Get the probability for the first (and only) sample
    
    # Convert probability to class label with improved threshold
    if probability >= 0.6:
        predicted_label = 'FAKE'
        confidence = probability
    else:
        predicted_label = 'REAL'
        confidence = 1 - probability
    
    return predicted_label, float(confidence), original_frames, frames_base64

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

@app.route('/api/predict', methods=['POST'])
def predict_deepfake():
    """Predict if uploaded video is real or fake"""
    try:
        print(f"📡 Received prediction request")
        print(f"📄 Request files: {list(request.files.keys())}")
        print(f"📄 Request form: {list(request.form.keys())}")
        
        # Check if file is present
        if 'video' not in request.files:
            print("❌ No 'video' key in request.files")
            return jsonify({'error': 'No video file provided'}), 400
        
        file = request.files['video']
        print(f"📁 File received: {file.filename}")
        print(f"📊 File size: {file.content_length if hasattr(file, 'content_length') else 'Unknown'}")
        
        if file.filename == '':
            print("❌ Empty filename")
            return jsonify({'error': 'No file selected'}), 400
        
        # Save uploaded file temporarily first to check file size
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            temp_file_path = temp_file.name
            file.save(temp_file_path)
        
        # Check file size (50 MB limit)
        file_size = os.path.getsize(temp_file_path)
        print(f"📊 Actual file size: {file_size / (1024*1024):.2f} MB")
        
        if file_size > 50 * 1024 * 1024:  # 50 MB
            print(f"❌ File too large: {file_size / (1024*1024):.2f} MB > 50 MB")
            os.unlink(temp_file_path)  # Clean up
            return jsonify({'error': 'File size exceeds 50 MB limit!'}), 400
        
        # Check file extension
        if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            print(f"❌ Invalid file extension: {file.filename}")
            os.unlink(temp_file_path)  # Clean up
            return jsonify({'error': 'Invalid file format. Please upload a video file.'}), 400
        
        print(f"✅ File validation passed: {file.filename}")
        print(f"🔄 Starting prediction...")
        
        # Make prediction
        predicted_label, confidence, extracted_frames, frames_base64 = predict_video(model, temp_file_path)
        
        # Clean up temporary file
        os.unlink(temp_file_path)
        
        # Prepare response
        frame_count = len(extracted_frames)
        
        return jsonify({
            'prediction': predicted_label,
            'confidence': confidence,
            'frames_extracted': frame_count,
            'frames': frames_base64,
            'message': f'Video predicted as {predicted_label} with {confidence:.2%} confidence'
        })
    
    except Exception as e:
        # Clean up temp file if it exists
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Get model information"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    return jsonify({
        'model_loaded': True,
        'input_shape': str(model.input_shape) if model else None,
        'output_shape': str(model.output_shape) if model else None,
        'max_frames': MAX_FRAMES,
        'image_size': IMG_SIZE
    })

# Error handler for file size limit
@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e):
    return jsonify({'error': 'File size exceeds 50 MB limit!'}), 413

if __name__ == '__main__':
    print("Starting Flask server...")
    print(f"Model loaded: {model is not None}")
    print(f"Max file size: {app.config['MAX_CONTENT_LENGTH'] / (1024*1024):.0f} MB")
    app.run(debug=True, host='0.0.0.0', port=5000)
