from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from sqlalchemy.orm import DeclarativeBase

# Define base model
class Base(DeclarativeBase):
    pass

# Initialize SQLAlchemy with our custom model class
db = SQLAlchemy(model_class=Base)

class User(db.Model):
    """
    User model for authentication.
    """
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationship with Analysis
    analyses = db.relationship('Analysis', backref='user', lazy=True)
    
    def __repr__(self):
        return f'<User {self.username}>'

class Analysis(db.Model):
    """
    Enhanced model for storing video analysis results with additional metrics.
    """
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    original_filename = db.Column(db.String(255), nullable=True)
    file_path = db.Column(db.String(255), nullable=False)
    file_size = db.Column(db.Integer, nullable=True)  # Size in bytes
    
    # Basic analysis results
    real_probability = db.Column(db.Float, nullable=False)
    fake_probability = db.Column(db.Float, nullable=False)
    is_fake = db.Column(db.Boolean, nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    processing_time = db.Column(db.Float, nullable=False)
    analyzed_frames = db.Column(db.Integer, nullable=False)
    
    # Enhanced video analysis data
    total_frames = db.Column(db.Integer, nullable=True)
    fps = db.Column(db.Float, nullable=True)
    resolution_width = db.Column(db.Integer, nullable=True)
    resolution_height = db.Column(db.Integer, nullable=True)
    prediction_std = db.Column(db.Float, nullable=True)  # Standard deviation of predictions
    scene_changes = db.Column(db.Integer, nullable=True)  # Number of scene changes detected
    
    # Error information
    error_message = db.Column(db.Text, nullable=True)  # For storing any error that occurred
    
    # Metadata
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    
    def __repr__(self):
        return f'<Analysis {self.id} - {"Fake" if self.is_fake else "Real"}>'
    
    def to_dict(self):
        """
        Convert the analysis to a dictionary for JSON serialization including enhanced metrics.
        """
        resolution = None
        if self.resolution_width and self.resolution_height:
            resolution = f"{self.resolution_width}x{self.resolution_height}"
            
        return {
            'id': self.id,
            'filename': self.filename,
            'original_filename': self.original_filename,
            'real_probability': self.real_probability,
            'fake_probability': self.fake_probability,
            'is_fake': self.is_fake,
            'confidence': self.confidence,
            'processing_time': self.processing_time,
            'analyzed_frames': self.analyzed_frames,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            
            # Enhanced metrics
            'total_frames': self.total_frames,
            'fps': self.fps,
            'resolution': resolution,
            'prediction_std': self.prediction_std,
            'scene_changes': self.scene_changes,
            'error_message': self.error_message,
            'file_size': self.file_size
        }
