"""Models for autonomous system monitoring and optimization"""
import logging
from datetime import datetime
from flask_login import UserMixin
from extensions import db

logger = logging.getLogger(__name__)

# Define signal_provider as association table
signal_provider = db.Table('signal_provider',
    db.Column('user_id', db.Integer, db.ForeignKey('user.id'), primary_key=True),
    db.Column('trade_signal_id', db.Integer, db.ForeignKey('trade_signal.id'), primary_key=True)
)

class User(UserMixin, db.Model):
    """User model for authentication and profile management"""
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)
    role = db.Column(db.String(20), default='user')
    reputation_score = db.Column(db.Float, default=5.0)
    chat_history = db.Column(db.JSON, default=list)
    total_signals = db.Column(db.Integer, default=0)
    successful_signals = db.Column(db.Integer, default=0)
    avg_profit = db.Column(db.Float, default=0.0)
    win_rate = db.Column(db.Float, default=0.0)
    risk_rating = db.Column(db.Float, default=5.0)
    is_verified = db.Column(db.Boolean, default=False)
    specialties = db.Column(db.JSON, default=list)  # e.g., ['BSC', 'Solana', 'Memecoins']
    followers = db.Column(db.Integer, default=0)
    provider_status = db.Column(db.String(20), default='active')
    # New gamification fields
    level = db.Column(db.Integer, default=1)
    experience_points = db.Column(db.Integer, default=0)
    achievements = db.Column(db.JSON, default=list)  # List of earned achievements
    weekly_rank = db.Column(db.Integer)
    monthly_rank = db.Column(db.Integer)
    badges = db.Column(db.JSON, default=list)  # e.g., ['top_performer', 'accuracy_master']
    streak_days = db.Column(db.Integer, default=0)  # Consecutive days with successful signals
    total_pnl = db.Column(db.Float, default=0.0)  # Total profit/loss

    # Signal marketplace relationships with explicit foreign keys
    signals_published = db.relationship('TradeSignal', 
                                    secondary=signal_provider,
                                    backref='publisher',
                                    lazy='dynamic')
    signal_subscriptions = db.relationship('SignalSubscription',
                                        foreign_keys='SignalSubscription.user_id',
                                        backref='subscriber',
                                        lazy='dynamic')
    provided_signals = db.relationship('SignalSubscription',
                                   foreign_keys='SignalSubscription.provider_id',
                                   backref='provider',
                                   lazy='dynamic')

    def update_provider_metrics(self, signal_result):
        """Update provider metrics based on signal performance"""
        try:
            self.total_signals += 1
            if signal_result.get('success'):
                self.successful_signals += 1
                self.experience_points += 10  # Points for successful signal
                self.streak_days += 1

                # Update total PnL
                self.total_pnl += signal_result.get('profit', 0)

                # Level up system
                if self.experience_points >= (self.level * 100):  # Simple level up formula
                    self.level += 1
                    # Add achievement for leveling up
                    if 'level_ups' not in self.achievements:
                        self.achievements.append('level_ups')
            else:
                self.streak_days = 0  # Reset streak on unsuccessful signal
                self.experience_points = max(0, self.experience_points - 5)  # Penalty for failed signal

            self.win_rate = (self.successful_signals / self.total_signals) * 100
            self.reputation_score = min(10.0, self.reputation_score + (0.1 if signal_result.get('success') else -0.2))
            self.avg_profit = ((self.avg_profit * (self.total_signals - 1)) + signal_result.get('profit', 0)) / self.total_signals

            # Update badges based on performance
            self._update_badges()

            db.session.commit()
            return True

        except Exception as e:
            logger.error(f"Error updating provider metrics: {str(e)}")
            db.session.rollback()
            return False

    def _update_badges(self):
        """Update user badges based on performance metrics"""
        new_badges = []

        # Accuracy badge
        if self.win_rate >= 70 and self.total_signals >= 20:
            new_badges.append('accuracy_master')

        # Volume badge
        if self.total_signals >= 100:
            new_badges.append('volume_trader')

        # Profit badge
        if self.total_pnl >= 1000:
            new_badges.append('profit_maker')

        # Streak badge
        if self.streak_days >= 7:
            new_badges.append('consistent_performer')

        # Level badge
        if self.level >= 10:
            new_badges.append('expert_analyst')

        # Update badges without duplicates
        self.badges = list(set(self.badges + new_badges))

    def to_provider_dict(self):
        return {
            'id': self.id,
            'username': self.username,
            'reputation_score': self.reputation_score,
            'total_signals': self.total_signals,
            'successful_signals': self.successful_signals,
            'win_rate': self.win_rate,
            'avg_profit': self.avg_profit,
            'risk_rating': self.risk_rating,
            'is_verified': self.is_verified,
            'status': self.provider_status,
            'specialties': self.specialties,
            'followers': self.followers,
            'created_at': self.created_at.isoformat(),
            # Add gamification data
            'level': self.level,
            'experience_points': self.experience_points,
            'badges': self.badges,
            'achievements': self.achievements,
            'streak_days': self.streak_days,
            'weekly_rank': self.weekly_rank,
            'monthly_rank': self.monthly_rank,
            'total_pnl': self.total_pnl
        }

class SystemMetrics(db.Model):
    """Track system performance metrics over time"""
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    cpu_usage = db.Column(db.Float)
    memory_usage = db.Column(db.Float)
    eth_gas_price = db.Column(db.Float)
    sol_network_tps = db.Column(db.Float)  # Solana network TPS
    sol_slot_time = db.Column(db.Float)    # Solana slot processing time
    llm_latency = db.Column(db.Float)
    success_rate = db.Column(db.Float)

    def to_dict(self):
        return {
            'timestamp': self.timestamp.isoformat(),
            'metrics': {
                'cpu_usage': self.cpu_usage,
                'memory_usage': self.memory_usage,
                'eth_gas_price': self.eth_gas_price,
                'sol_network_tps': self.sol_network_tps,
                'sol_slot_time': self.sol_slot_time,
                'llm_latency': self.llm_latency,
                'success_rate': self.success_rate
            }
        }

class OptimizationEvent(db.Model):
    """Track system optimization events and their impact"""
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    component = db.Column(db.String(50), nullable=False)
    action_taken = db.Column(db.Text, nullable=False)
    parameters_before = db.Column(db.JSON)
    parameters_after = db.Column(db.JSON)
    impact_score = db.Column(db.Float)
    success = db.Column(db.Boolean, default=True)

    def to_dict(self):
        return {
            'timestamp': self.timestamp.isoformat(),
            'component': self.component,
            'action_taken': self.action_taken,
            'parameters_before': self.parameters_before,
            'parameters_after': self.parameters_after,
            'impact_score': self.impact_score,
            'success': self.success
        }

class Trade(db.Model):
    """Record of executed trades with enhanced tracking"""
    id = db.Column(db.Integer, primary_key=True)
    token_address = db.Column(db.String(64))
    token_symbol = db.Column(db.String(10), nullable=False)
    chain = db.Column(db.String(20), nullable=False)
    price = db.Column(db.Float, nullable=False)
    amount = db.Column(db.Float, nullable=False)
    trade_type = db.Column(db.String(20), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    profit = db.Column(db.Float, default=0.0)
    potential_score = db.Column(db.Float)
    stop_loss = db.Column(db.Float)
    take_profit = db.Column(db.Float)
    risk_score = db.Column(db.Float)
    position_size_usd = db.Column(db.Float)
    slippage = db.Column(db.Float)
    tx_hash = db.Column(db.String(128))
    strategy_notes = db.Column(db.Text)
    strategy_id = db.Column(db.Integer, db.ForeignKey('custom_strategy.id'))
    strategy_entry_score = db.Column(db.Float)
    strategy_exit_score = db.Column(db.Float)
    strategy_position_size = db.Column(db.Float)
    strategy_risk_level = db.Column(db.String(10))
    entry_indicators = db.Column(db.JSON)
    exit_indicators = db.Column(db.JSON)
    status = db.Column(db.String(20), default='pending')

    def to_dict(self):
        return {
            'id': self.id,
            'token_address': self.token_address,
            'token_symbol': self.token_symbol,
            'chain': self.chain,
            'price': self.price,
            'amount': self.amount,
            'trade_type': self.trade_type,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'profit': self.profit,
            'potential_score': self.potential_score,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'risk_score': self.risk_score,
            'position_size_usd': self.position_size_usd,
            'slippage': self.slippage,
            'tx_hash': self.tx_hash,
            'status': self.status,
            'strategy': {
                'id': self.strategy_id,
                'entry_score': self.strategy_entry_score,
                'exit_score': self.strategy_exit_score,
                'position_size': self.strategy_position_size,
                'risk_level': self.strategy_risk_level,
                'notes': self.strategy_notes
            }
        }

class TokenMetrics(db.Model):
    """Token metrics and analysis data with enhanced Solana memecoin tracking"""
    id = db.Column(db.Integer, primary_key=True)
    address = db.Column(db.String(64), nullable=False)
    chain = db.Column(db.String(20), nullable=False, default='solana')
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    launch_timestamp = db.Column(db.DateTime)
    price = db.Column(db.Float)
    volume = db.Column(db.Float)
    liquidity = db.Column(db.Float)
    initial_liquidity = db.Column(db.Float)
    holder_count = db.Column(db.Integer)
    safety_score = db.Column(db.Float)  # SolanaSniffer safety score
    liquidity_locked = db.Column(db.Boolean)  # Whether liquidity is locked
    mint_enabled = db.Column(db.Boolean)  # Whether minting is enabled
    contract_verified = db.Column(db.Boolean)  # Whether contract is verified
    raydium_pool_exists = db.Column(db.Boolean)  # Whether Raydium pool exists
    holder_analysis = db.Column(db.JSON)
    contract_analysis = db.Column(db.JSON)
    liquidity_analysis = db.Column(db.JSON)
    risk_score = db.Column(db.Float)
    potential_score = db.Column(db.Float)
    rsi = db.Column(db.Float)
    macd = db.Column(db.Float)
    bollinger_bands = db.Column(db.JSON)
    moving_averages = db.Column(db.JSON)
    status = db.Column(db.String(20), default='active')
    launch_platform = db.Column(db.String(20))  # 'pumpfun' or 'gmgn'
    social_mentions = db.Column(db.Integer)
    influencer_mentions = db.Column(db.Integer)
    telegram_sentiment = db.Column(db.Float)

    __table_args__ = (db.UniqueConstraint('address', 'chain', name='unique_token_chain'),)

    def to_dict(self):
        return {
            'address': self.address,
            'chain': self.chain,
            'timestamp': self.timestamp.isoformat(),
            'price': self.price,
            'volume': self.volume,
            'liquidity': self.liquidity,
            'safety_metrics': {
                'safety_score': self.safety_score,
                'liquidity_locked': self.liquidity_locked,
                'mint_enabled': self.mint_enabled,
                'contract_verified': self.contract_verified,
                'raydium_pool_exists': self.raydium_pool_exists
            },
            'social_metrics': {
                'social_mentions': self.social_mentions,
                'influencer_mentions': self.influencer_mentions,
                'telegram_sentiment': self.telegram_sentiment
            },
            'analysis': {
                'risk_score': self.risk_score,
                'potential_score': self.potential_score,
                'holder_analysis': self.holder_analysis,
                'contract_analysis': self.contract_analysis,
                'technical_indicators': {
                    'rsi': self.rsi,
                    'macd': self.macd,
                    'bollinger_bands': self.bollinger_bands,
                    'moving_averages': self.moving_averages
                }
            }
        }

class CustomStrategy(db.Model):
    """Custom trading strategy configuration"""
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False, unique=True)
    chain = db.Column(db.String(20), nullable=False)
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    entry_conditions = db.Column(db.JSON, nullable=False)
    exit_conditions = db.Column(db.JSON, nullable=False)
    position_size_config = db.Column(db.JSON, nullable=False)
    risk_parameters = db.Column(db.JSON, nullable=False)
    total_trades = db.Column(db.Integer, default=0)
    successful_trades = db.Column(db.Integer, default=0)
    total_profit = db.Column(db.Float, default=0.0)
    max_drawdown = db.Column(db.Float, default=0.0)
    win_rate = db.Column(db.Float, default=0.0)

    def validate_strategy(self):
        required_fields = ['indicators', 'thresholds', 'timeframes']
        if not all(field in self.entry_conditions for field in required_fields):
            return False, "Missing required entry condition fields"
        if not all(field in self.exit_conditions for field in required_fields):
            return False, "Missing required exit condition fields"
        required_position_fields = ['base_size', 'max_size', 'size_increment']
        if not all(field in self.position_size_config for field in required_position_fields):
            return False, "Missing required position sizing fields"
        required_risk_fields = ['max_risk_per_trade', 'max_daily_risk', 'stop_loss_type']
        if not all(field in self.risk_parameters for field in required_risk_fields):
            return False, "Missing required risk parameter fields"
        return True, "Strategy configuration is valid"

    def update_performance(self, trade_result):
        self.total_trades += 1
        if trade_result['profit'] > 0:
            self.successful_trades += 1
        self.total_profit += trade_result['profit']
        self.win_rate = (self.successful_trades / self.total_trades) * 100 if self.total_trades > 0 else 0.0
        if trade_result.get('drawdown', 0) > self.max_drawdown:
            self.max_drawdown = trade_result['drawdown']

    def get_position_size(self, available_capital, market_conditions):
        try:
            base_size = self.position_size_config['base_size']
            max_size = self.position_size_config['max_size']
            size_increment = self.position_size_config['size_increment']
            position_size = base_size * available_capital
            if market_conditions.get('trend_strength', 0) > 0.7:
                position_size *= (1 + size_increment)
            if market_conditions.get('volatility', 0) > 0.5:
                position_size *= (1 - size_increment)
            return min(position_size, max_size * available_capital)
        except Exception as e:
            return base_size * available_capital

    def adjust_parameters_based_on_performance(self):
        """Automatically adjust strategy parameters based on historical performance"""
        try:
            if self.total_trades < 10:
                return False, "Not enough trades for parameter adjustment"

            avg_profit = self.total_profit / self.total_trades if self.total_trades > 0 else 0
            win_rate = self.win_rate / 100  # Convert to decimal

            if win_rate > 0.7 and avg_profit > 0:
                self.position_size_config['base_size'] = min(
                    self.position_size_config['base_size'] * 1.1,
                    self.position_size_config['max_size']
                )
            elif win_rate < 0.4 or avg_profit < 0:
                self.position_size_config['base_size'] *= 0.9

            if self.max_drawdown > self.risk_parameters['max_risk_per_trade']:
                self.risk_parameters['max_risk_per_trade'] *= 0.9
                if 'stop_loss_percentage' in self.risk_parameters: #check if the key exists
                    self.risk_parameters['stop_loss_percentage'] *= 1.1
            elif self.max_drawdown < self.risk_parameters['max_risk_per_trade'] * 0.5:
                self.risk_parameters['max_risk_per_trade'] = min(
                    self.risk_parameters['max_risk_per_trade'] * 1.1,
                    0.05  # Maximum 5% risk per trade
                )

            if win_rate < 0.5:
                for threshold in self.entry_conditions.get('thresholds', {}).values():
                    if 'min' in threshold:
                        threshold['min'] *= 1.1
                    if 'max' in threshold:
                        threshold['max'] *= 0.9

            db.session.commit()
            return True, "Strategy parameters adjusted successfully"

        except Exception as e:
            logger.error(f"Error adjusting strategy parameters: {str(e)}")
            db.session.rollback()
            return False, f"Error adjusting parameters: {str(e)}"

    def analyze_performance(self):
        """Analyze strategy performance and generate insights"""
        try:
            if self.total_trades < 5:
                return {
                    'status': 'insufficient_data',
                    'message': 'Not enough trades for meaningful analysis'
                }

            performance_metrics = {
                'win_rate': self.win_rate,
                'avg_profit': self.total_profit / self.total_trades if self.total_trades > 0 else 0,
                'max_drawdown': self.max_drawdown,
                'risk_reward_ratio': abs(self.total_profit / (self.max_drawdown if self.max_drawdown != 0 else 1)),
                'total_trades': self.total_trades,
                'is_profitable': self.total_profit > 0
            }

            if performance_metrics['win_rate'] > 60 and performance_metrics['risk_reward_ratio'] > 2:
                performance_metrics['rating'] = 'Excellent'
            elif performance_metrics['win_rate'] > 50 and performance_metrics['risk_reward_ratio'] > 1.5:
                performance_metrics['rating'] = 'Good'
            elif performance_metrics['win_rate'] > 40:
                performance_metrics['rating'] = 'Fair'
            else:
                performance_metrics['rating'] = 'Poor'

            recommendations = []
            if performance_metrics['win_rate'] < 50:
                recommendations.append("Consider increasing entry threshold stringency")
            if performance_metrics['max_drawdown'] > self.risk_parameters['max_risk_per_trade'] * 2:
                recommendations.append("Review stop-loss levels and position sizing")
            if performance_metrics['risk_reward_ratio'] < 1:
                recommendations.append("Adjust take-profit levels for better risk-reward ratio")

            performance_metrics['recommendations'] = recommendations
            return performance_metrics

        except Exception as e:
            logger.error(f"Error analyzing strategy performance: {str(e)}")
            return {
                'status': 'error',
                'message': f"Error during analysis: {str(e)}"
            }

class TradeSignal(db.Model):
    """Community trading signals with performance tracking"""
    id = db.Column(db.Integer, primary_key=True)
    provider_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    token_address = db.Column(db.String(64), nullable=False)
    chain = db.Column(db.String(20), nullable=False)
    signal_type = db.Column(db.String(20), nullable=False)  # 'BUY' or 'SELL'
    entry_price = db.Column(db.Float)
    target_price = db.Column(db.Float)
    stop_loss = db.Column(db.Float)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    expiry = db.Column(db.DateTime)
    status = db.Column(db.String(20), default='active')  # active, expired, completed
    confidence_score = db.Column(db.Float)
    risk_level = db.Column(db.String(20))
    technical_indicators = db.Column(db.JSON)
    verification_status = db.Column(db.String(20), default='pending')
    actual_profit = db.Column(db.Float)
    success_rate = db.Column(db.Float)
    verified_by_system = db.Column(db.Boolean, default=False)
    description = db.Column(db.Text)
    tags = db.Column(db.JSON)
    likes_count = db.Column(db.Integer, default=0)

    def to_dict(self):
        return {
            'id': self.id,
            'token_address': self.token_address,
            'chain': self.chain,
            'signal_type': self.signal_type,
            'entry_price': self.entry_price,
            'target_price': self.target_price,
            'stop_loss': self.stop_loss,
            'timestamp': self.timestamp.isoformat(),
            'expiry': self.expiry.isoformat() if self.expiry else None,
            'status': self.status,
            'confidence_score': self.confidence_score,
            'risk_level': self.risk_level,
            'verification_status': self.verification_status,
            'success_rate': self.success_rate,
            'description': self.description,
            'likes_count': self.likes_count,
            'provider': self.publisher[0].username if self.publisher else None
        }

class SignalSubscription(db.Model):
    """User subscriptions to signal providers"""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    provider_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    subscription_type = db.Column(db.String(20), default='free')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    expires_at = db.Column(db.DateTime)
    is_active = db.Column(db.Boolean, default=True)

    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'provider_id': self.provider_id,
            'subscription_type': self.subscription_type,
            'created_at': self.created_at.isoformat(),
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'is_active': self.is_active
        }

class KOLMetrics(db.Model):
    """Track KOL (Key Opinion Leader) performance metrics"""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(64), unique=True, nullable=False)  # Telegram user ID
    username = db.Column(db.String(64))
    average_volume = db.Column(db.Float, default=0.0)  # Average trading volume of calls
    success_rate = db.Column(db.Float, default=0.0)  # Overall success rate
    total_calls = db.Column(db.Integer, default=0)  # Total number of calls
    successful_calls = db.Column(db.Integer, default=0)  # Number of successful calls
    avg_gain_multiple = db.Column(db.Float, default=0.0)  # Average gain multiple across all calls
    total_2x_calls = db.Column(db.Integer, default=0)  # Number of calls that reached 2x
    best_gain_multiple = db.Column(db.Float, default=0.0)  # Best gain multiple achieved
    influence_score = db.Column(db.Float, default=1.0)  # KOL influence rating
    last_updated = db.Column(db.DateTime, default=datetime.utcnow)
    specialties = db.Column(db.JSON, default=list)  # e.g., ['new_launches', 'memecoins']
    risk_rating = db.Column(db.Float, default=5.0)  # Risk level of calls (1-10)
    verified = db.Column(db.Boolean, default=False)

    def update_metrics(self, call_result):
        """Update KOL metrics based on new call performance"""
        try:
            self.total_calls += 1
            if call_result.get('success'):
                self.successful_calls += 1
                gain_multiple = call_result.get('gain_multiple', 0)

                # Update 2x calls tracking
                if gain_multiple >= 2.0:
                    self.total_2x_calls += 1

                # Update best gain multiple
                if gain_multiple > self.best_gain_multiple:
                    self.best_gain_multiple = gain_multiple

                # Update average gain multiple
                self.avg_gain_multiple = (
                    (self.avg_gain_multiple * (self.successful_calls - 1) + gain_multiple) 
                    / self.successful_calls
                )

            # Update success rate
            self.success_rate = (self.successful_calls / self.total_calls) * 100

            # Update average volume
            new_volume = call_result.get('volume', 0)
            self.average_volume = (
                (self.average_volume * (self.total_calls - 1) + new_volume) 
                / self.total_calls
            )

            self.last_updated = datetime.utcnow()
            db.session.commit()
            return True

        except Exception as e:
            logger.error(f"Error updating KOL metrics: {str(e)}")
            db.session.rollback()
            return False

    def to_dict(self):
        """Convert KOL metrics to dictionary"""
        return {
            'user_id': self.user_id,
            'username': self.username,
            'metrics': {
                'average_volume': self.average_volume,
                'success_rate': self.success_rate,
                'total_calls': self.total_calls,
                'successful_calls': self.successful_calls,
                'avg_gain_multiple': self.avg_gain_multiple,
                'total_2x_calls': self.total_2x_calls,
                'best_gain_multiple': self.best_gain_multiple,
                'influence_score': self.influence_score
            },
            'specialties': self.specialties,
            'risk_rating': self.risk_rating,
            'verified': self.verified,
            'last_updated': self.last_updated.isoformat()
        }

class TelegramSentiment(db.Model):
    """Store sentiment analysis results from Telegram messages"""
    id = db.Column(db.Integer, primary_key=True)
    token_address = db.Column(db.String(64), nullable=False)
    chain = db.Column(db.String(20), nullable=False)
    group_name = db.Column(db.String(64), nullable=False)
    message_count = db.Column(db.Integer, default=0)
    positive_count = db.Column(db.Integer, default=0)
    negative_count = db.Column(db.Integer, default=0)
    neutral_count = db.Column(db.Integer, default=0)
    sentiment_score = db.Column(db.Float)
    spydefi_rating = db.Column(db.String(20))
    top_keywords = db.Column(db.JSON)
    kol_weighted_score = db.Column(db.Float, default=0.0)
    is_new_launch = db.Column(db.Boolean, default=False)
    last_updated = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'token_address': self.token_address,
            'chain': self.chain,
            'group_name': self.group_name,
            'metrics': {
                'message_count': self.message_count,
                'positive_count': self.positive_count,
                'negative_count': self.negative_count,
                'neutral_count': self.neutral_count,
                'sentiment_score': self.sentiment_score,
                'kol_weighted_score': self.kol_weighted_score
            },
            'spydefi_rating': self.spydefi_rating,
            'top_keywords': self.top_keywords,
            'is_new_launch': self.is_new_launch,
            'last_updated': self.last_updated.isoformat()
        }

class TelegramMessages(db.Model):
    """Store individual Telegram messages with KOL attribution"""
    id = db.Column(db.Integer, primary_key=True)
    token_address = db.Column(db.String(64), nullable=False)
    chain = db.Column(db.String(20), nullable=False)
    group_name = db.Column(db.String(64), nullable=False)
    message_text = db.Column(db.Text, nullable=False)
    sentiment = db.Column(db.Float)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    user_influence_score = db.Column(db.Float, default=1.0)
    kol_id = db.Column(db.String(64))  # Link to KOLMetrics
    kol_volume = db.Column(db.Float)  # KOL's average trading volume
    kol_success_rate = db.Column(db.Float)  # KOL's success rate
    kol_avg_gain = db.Column(db.Float)  # KOL's average gain multiple
    kol_total_2x_calls = db.Column(db.Integer)  # KOL's total 2x successful calls
    is_new_launch = db.Column(db.Boolean, default=False)

    def to_dict(self):
        return {
            'id': self.id,
            'token_address': self.token_address,
            'chain': self.chain,
            'group_name': self.group_name,
            'message': {
                'text': self.message_text,
                'sentiment': self.sentiment,
                'timestamp': self.timestamp.isoformat()
            },
            'kol_data': {
                'id': self.kol_id,
                'influence_score': self.user_influence_score,
                'volume': self.kol_volume,
                'success_rate': self.kol_success_rate,
                'avg_gain': self.kol_avg_gain,
                'total_2x_calls': self.kol_total_2x_calls
            },
            'is_new_launch': self.is_new_launch
        }