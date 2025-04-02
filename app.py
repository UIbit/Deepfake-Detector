import os
import uuid
import logging
from flask import Flask, render_template, request, redirect, url_for, flash, session
import cv2
import numpy as np
from werkzeug.utils import secure_filename
from detector import DeepfakeDetector

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "deepfake_detector_secret")

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'wmv', 'flv', 'mkv'}
MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB limit

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Initialize the deepfake detector
detector = DeepfakeDetector()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if a file was submitted
    if 'video' not in request.files:
        flash('No file part', 'danger')
        return redirect(request.url)
    
    file = request.files['video']
    
    # Check if user submitted an empty form
    if file.filename == '':
        flash('No video selected', 'danger')
        return redirect(url_for('index'))
    
    # Check if the file is allowed
    if not allowed_file(file.filename):
        flash(f'Invalid file type. Allowed types are: {", ".join(ALLOWED_EXTENSIONS)}', 'danger')
        return redirect(url_for('index'))
    
    try:
        # Generate a unique filename to prevent overwriting
        filename = str(uuid.uuid4()) + '_' + secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process the video
        result = detector.analyze_video(filepath)
        
        # Store results in session
        session['results'] = {
            'filename': filename,
            'filepath': filepath,
            'real_probability': result['real_probability'],
            'fake_probability': result['fake_probability'],
            'is_fake': result['is_fake'],
            'confidence': result['confidence'],
            'processing_time': result['processing_time'],
            'analyzed_frames': result['analyzed_frames']
        }
        
        return redirect(url_for('results'))
    
    except Exception as e:
        logging.error(f"Error processing video: {str(e)}")
        flash(f'Error processing video: {str(e)}', 'danger')
        return redirect(url_for('index'))

@app.route('/results')
def results():
    if 'results' not in session:
        flash('No results to display. Please upload a video first.', 'warning')
        return redirect(url_for('index'))
    
    return render_template('results.html', results=session['results'])

@app.errorhandler(413)
def request_entity_too_large(error):
    flash(f'File too large. Maximum size is {MAX_CONTENT_LENGTH/(1024*1024)}MB', 'danger')
    return redirect(url_for('index'))

# Add cleanup for temporary files (could be improved with a scheduled task)
@app.teardown_appcontext
def cleanup_old_files(exception):
    # This is a simple implementation; a more robust solution would use a scheduled task
    import glob
    import time
    
    # Remove files older than 1 hour
    current_time = time.time()
    one_hour_ago = current_time - 3600
    
    for file in glob.glob(os.path.join(app.config['UPLOAD_FOLDER'], '*')):
        if os.path.getmtime(file) < one_hour_ago:
            try:
                os.remove(file)
                logging.debug(f"Removed old file: {file}")
            except Exception as e:
                logging.error(f"Error removing file {file}: {str(e)}")
