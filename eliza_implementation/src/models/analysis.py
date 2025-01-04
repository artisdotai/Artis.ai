"""
Analysis models for Eliza Framework
"""
from .base import db, TimestampMixin

class TokenAnalysis(db.Model, TimestampMixin):
    """Token analysis and metrics model"""
    __tablename__ = 'token_analysis'

    id = db.Column(db.Integer, primary_key=True)
    token_address = db.Column(db.String(255), nullable=False)
    chain = db.Column(db.String(50), nullable=False)
    price = db.Column(db.Float)
    market_cap = db.Column(db.Float)
    liquidity = db.Column(db.Float)
    volume_24h = db.Column(db.Float)
    holder_count = db.Column(db.Integer)
    risk_score = db.Column(db.Float)
    sentiment_score = db.Column(db.Float)
    technical_rating = db.Column(db.Text)  # Changed from String(255) to Text for longer content

    __table_args__ = (
        db.UniqueConstraint('token_address', 'chain', name='unique_token_chain'),
    )

class SentimentAnalysis(db.Model, TimestampMixin):
    """Social sentiment tracking model"""
    __tablename__ = 'sentiment_analysis'

    id = db.Column(db.Integer, primary_key=True)
    source = db.Column(db.String(100), nullable=False)
    content_hash = db.Column(db.String(64))
    sentiment_score = db.Column(db.Float)
    influence_score = db.Column(db.Float)
    mentions_count = db.Column(db.Integer)
    engagement_rate = db.Column(db.Float)

    __table_args__ = (
        db.UniqueConstraint('content_hash', name='unique_content_hash'),
    )