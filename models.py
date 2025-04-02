# This file would normally contain SQLAlchemy models if we were using a database
# For this application, we're not using a database, but we keep this file
# for consistency with the Flask project structure

class AnalysisResult:
    """
    Model representing the result of a deepfake analysis.
    This is a simple class to structure our analysis results.
    """
    def __init__(self, 
                 filename, 
                 real_probability, 
                 fake_probability, 
                 is_fake, 
                 confidence, 
                 processing_time, 
                 analyzed_frames):
        self.filename = filename
        self.real_probability = real_probability
        self.fake_probability = fake_probability
        self.is_fake = is_fake
        self.confidence = confidence
        self.processing_time = processing_time
        self.analyzed_frames = analyzed_frames
