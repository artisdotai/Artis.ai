"""
Base model configurations for Eliza Framework
"""
from ..database import db

class TimestampMixin:
    """Timestamp mixin for models"""
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())
    updated_at = db.Column(db.DateTime, 
                          default=db.func.current_timestamp(),
                          onupdate=db.func.current_timestamp())