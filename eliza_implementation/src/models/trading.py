"""
Trading models for Eliza Framework
"""
from .base import db, TimestampMixin

class Trade(db.Model, TimestampMixin):
    """Trade model for tracking cryptocurrency trades"""
    __tablename__ = 'trades'

    id = db.Column(db.Integer, primary_key=True)
    chain = db.Column(db.String(50), nullable=False)
    token_address = db.Column(db.String(255), nullable=False)
    token_symbol = db.Column(db.String(50))
    entry_price = db.Column(db.Float, nullable=False)
    exit_price = db.Column(db.Float)
    position_size = db.Column(db.Float, nullable=False)
    status = db.Column(db.String(50), default='open')
    risk_score = db.Column(db.Float)
    profit_loss = db.Column(db.Float)
    strategy = db.Column(db.String(100))

    __table_args__ = (
        db.UniqueConstraint('chain', 'token_address', name='unique_trade_token'),
    )

class ChainMetrics(db.Model, TimestampMixin):
    """Chain-specific metrics tracking"""
    __tablename__ = 'chain_metrics'

    id = db.Column(db.Integer, primary_key=True)
    chain = db.Column(db.String(50), nullable=False)
    total_liquidity = db.Column(db.Float)
    daily_volume = db.Column(db.Float)
    active_pairs = db.Column(db.Integer)
    gas_price = db.Column(db.Float)
    status = db.Column(db.String(50))

    __table_args__ = (
        db.UniqueConstraint('chain', name='unique_chain'),
    )