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
    Model for storing video analysis results.
    """
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    original_filename = db.Column(db.String(255), nullable=True)
    file_path = db.Column(db.String(255), nullable=False)
    file_size = db.Column(db.Integer, nullable=True)  # Size in bytes
    
    # Analysis results
    real_probability = db.Column(db.Float, nullable=False)
    fake_probability = db.Column(db.Float, nullable=False)
    is_fake = db.Column(db.Boolean, nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    processing_time = db.Column(db.Float, nullable=False)
    analyzed_frames = db.Column(db.Integer, nullable=False)
    
    # Metadata
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    
    def __repr__(self):
        return f'<Analysis {self.id} - {"Fake" if self.is_fake else "Real"}>'
    
    def to_dict(self):
        """
        Convert the analysis to a dictionary for JSON serialization.
        """
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
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
