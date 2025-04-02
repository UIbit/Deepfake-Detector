import os
import uuid
import logging
from flask import Flask, render_template, request, redirect, url_for, flash, session
import cv2
import numpy as np
from werkzeug.utils import secure_filename
from detector import DeepfakeDetector
from models import db, User, Analysis
from sqlalchemy.exc import SQLAlchemyError

# Configure logging
logging.basicConfig(level=logging.DEBUG, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "faster_than_lies_secret")
app.config['SUPPORTS_THEME_AUTO'] = True  # Flag for theme auto-detection support

# Increase request timeouts and buffer sizes
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB limit
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'wmv', 'flv', 'mkv'}

# Database configuration
# Format: postgresql://username:password@hostname:port/database_name
db_host = os.environ.get('PGHOST', 'localhost')
db_port = os.environ.get('PGPORT', '5432')
db_name = os.environ.get('PGDATABASE', 'postgres')
db_user = os.environ.get('PGUSER', 'postgres')
db_password = os.environ.get('PGPASSWORD', '')

database_url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
app.config['SQLALCHEMY_DATABASE_URI'] = database_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    'pool_recycle': 3600,  # Recycle connections after 1 hour
    'pool_pre_ping': True,  # Check connection validity before using it
    'connect_args': {'connect_timeout': 30}  # 30 second timeout on connections
}

logging.info(f"Database configured with host: {db_host}, port: {db_port}, database: {db_name}")

# Initialize the database
db.init_app(app)

# Configuration variables for access in functions
UPLOAD_FOLDER = app.config['UPLOAD_FOLDER']
ALLOWED_EXTENSIONS = app.config['ALLOWED_EXTENSIONS']
MAX_CONTENT_LENGTH = app.config['MAX_CONTENT_LENGTH']

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
logging.info(f"Upload directory configured: {UPLOAD_FOLDER}")

# Initialize the deepfake detector
detector = DeepfakeDetector()

# Create database tables
with app.app_context():
    db.create_all()
    logging.info("Database tables created")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/history')
def history():
    # Fetch the 10 most recent analyses
    try:
        recent_analyses = Analysis.query.order_by(Analysis.created_at.desc()).limit(10).all()
        return render_template('history.html', analyses=recent_analyses)
    except Exception as e:
        logging.error(f"Error retrieving analysis history: {str(e)}")
        flash(f'Error retrieving analysis history: {str(e)}', 'danger')
        return redirect(url_for('index'))

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        # Check if a file was submitted
        if 'video' not in request.files:
            flash('No file part', 'danger')
            return redirect(url_for('index'))
        
        file = request.files['video']
        
        # Check if user submitted an empty form
        if file.filename == '':
            flash('No video selected', 'danger')
            return redirect(url_for('index'))
        
        # Check if the file is allowed
        if not allowed_file(file.filename):
            flash(f'Invalid file type. Allowed types are: {", ".join(ALLOWED_EXTENSIONS)}', 'danger')
            return redirect(url_for('index'))
        
        # Generate a unique filename to prevent overwriting
        original_filename = secure_filename(file.filename)
        filename = str(uuid.uuid4()) + '_' + original_filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Ensure the upload directory exists
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        # Save the file safely
        file.save(filepath)
        
        # Verify the file was saved successfully
        if not os.path.exists(filepath):
            raise IOError("Failed to save the uploaded file")
        
        # Get file size
        file_size = os.path.getsize(filepath)
        logging.info(f"File saved: {filepath}, size: {file_size} bytes")
        
        # Process the video
        result = detector.analyze_video(filepath)
        logging.info(f"Analysis complete for {filename}")
        
        # Store in database
        analysis = Analysis(
            filename=filename,
            original_filename=original_filename,
            file_path=filepath,
            file_size=file_size,
            real_probability=float(result['real_probability']),
            fake_probability=float(result['fake_probability']),
            is_fake=bool(result['is_fake']),
            confidence=float(result['confidence']),
            processing_time=float(result['processing_time']),
            analyzed_frames=int(result['analyzed_frames'])
        )
        
        try:
            db.session.add(analysis)
            db.session.commit()
            logging.info(f"Analysis saved to database with ID: {analysis.id}")
            
            # Set the analysis ID in session for later retrieval
            session['analysis_id'] = analysis.id
            
        except SQLAlchemyError as db_error:
            db.session.rollback()
            logging.error(f"Database error: {str(db_error)}")
            # Continue without database storage if it fails
            
        # Store results in session as well for backward compatibility - convert boolean to string to avoid JSON serialization issues
        session['results'] = {
            'filename': filename,
            'filepath': filepath,
            'real_probability': float(result['real_probability']),
            'fake_probability': float(result['fake_probability']),
            'is_fake': 'true' if result['is_fake'] else 'false',
            'confidence': float(result['confidence']),
            'processing_time': float(result['processing_time']),
            'analyzed_frames': int(result['analyzed_frames'])
        }
        
        return redirect(url_for('results'))
    
    except Exception as e:
        logging.error(f"Error processing video: {str(e)}", exc_info=True)
        
        # Check if a file was uploaded and saved but processing failed
        filepath = locals().get('filepath')
        if filepath and os.path.exists(filepath):
            try:
                os.remove(filepath)
                logging.info(f"Removed failed upload: {filepath}")
            except Exception as cleanup_error:
                logging.error(f"Error removing failed upload: {str(cleanup_error)}")
                
        flash(f'Error processing video: {str(e)}', 'danger')
        return redirect(url_for('index'))

@app.route('/results')
def results():
    # Check if an analysis ID was passed as a parameter (from history page)
    analysis_id = request.args.get('analysis_id')
    
    if analysis_id:
        try:
            # Get the analysis from database with the provided ID
            analysis = Analysis.query.get(analysis_id)
            if analysis:
                # Convert to a dictionary format for the template
                results_dict = {
                    'filename': analysis.filename,
                    'filepath': analysis.file_path,
                    'real_probability': float(analysis.real_probability),
                    'fake_probability': float(analysis.fake_probability),
                    'is_fake': 'true' if analysis.is_fake else 'false',
                    'confidence': float(analysis.confidence),
                    'processing_time': float(analysis.processing_time),
                    'analyzed_frames': int(analysis.analyzed_frames),
                    'original_filename': analysis.original_filename,
                    'file_size': analysis.file_size,
                    'created_at': analysis.created_at
                }
                return render_template('results.html', results=results_dict, from_db=True)
        except Exception as e:
            logging.error(f"Error retrieving analysis from database: {str(e)}")
            flash(f'Error retrieving analysis: {str(e)}', 'danger')
            return redirect(url_for('history'))
    
    # Check if we have an analysis ID in the session (from upload)
    elif 'analysis_id' in session:
        # Fetch from database using ID
        session_analysis_id = session['analysis_id']
        try:
            analysis = Analysis.query.get(session_analysis_id)
            if analysis:
                # Convert to a dictionary format for the template
                results_dict = {
                    'filename': analysis.filename,
                    'filepath': analysis.file_path,
                    'real_probability': float(analysis.real_probability),
                    'fake_probability': float(analysis.fake_probability),
                    'is_fake': 'true' if analysis.is_fake else 'false',
                    'confidence': float(analysis.confidence),
                    'processing_time': float(analysis.processing_time),
                    'analyzed_frames': int(analysis.analyzed_frames),
                    'original_filename': analysis.original_filename,
                    'file_size': analysis.file_size,
                    'created_at': analysis.created_at
                }
                return render_template('results.html', results=results_dict, from_db=True)
        except Exception as e:
            logging.error(f"Error retrieving analysis from database: {str(e)}")
            # Fall back to session data if database retrieval fails
            
    # Fall back to session data
    if 'results' in session:
        return render_template('results.html', results=session['results'], from_db=False)
    
    # No results available
    flash('No results to display. Please upload a video first.', 'warning')
    return redirect(url_for('index'))

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
