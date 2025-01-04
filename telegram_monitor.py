"""Telegram channel monitoring for SpyDefi integration"""
import logging
from datetime import datetime
from models import TelegramSentiment, TelegramMessages, KOLMetrics, db
from spydefi_connector import SpydefiConnector

logger = logging.getLogger(__name__)

class TelegramMonitor:
    def __init__(self):
        """Initialize TelegramMonitor with enhanced error handling"""
        self.sentiment_cache = {}
        self.message_batch = []
        self.batch_size = 100
        self.spydefi = SpydefiConnector()
        self._circuit_breaker = {
            'failures': 0,
            'max_failures': 5,
            'is_open': False,
            'last_check': datetime.utcnow()
        }
        self.rate_limit = {
            'requests': 0,
            'max_requests': 100,
            'window_start': datetime.utcnow(),
            'window_size': 60  # seconds
        }

    def _check_rate_limit(self) -> bool:
        """Check if rate limit is exceeded"""
        now = datetime.utcnow()
        window_diff = (now - self.rate_limit['window_start']).total_seconds()

        if window_diff >= self.rate_limit['window_size']:
            # Reset window
            self.rate_limit['requests'] = 0
            self.rate_limit['window_start'] = now
            return True

        if self.rate_limit['requests'] >= self.rate_limit['max_requests']:
            logger.warning("Rate limit exceeded")
            return False

        self.rate_limit['requests'] += 1
        return True

    def _check_circuit_breaker(self) -> bool:
        """Check if circuit breaker is tripped"""
        now = datetime.utcnow()
        window_diff = (now - self._circuit_breaker['last_check']).total_seconds()

        if window_diff >= 300:  # Reset after 5 minutes
            self._circuit_breaker['failures'] = 0
            self._circuit_breaker['is_open'] = False
            self._circuit_breaker['last_check'] = now
            return True

        return not self._circuit_breaker['is_open']

    def process_message(self, message_data):
        """Process incoming Telegram message with SpyDefi integration"""
        try:
            if not self._check_rate_limit():
                logger.warning("Rate limit exceeded, dropping message")
                return False

            if not self._check_circuit_breaker():
                logger.warning("Circuit breaker open, dropping message")
                return False

            # Extract message data
            token_address = message_data.get('token_address')
            chain = message_data.get('chain', 'solana')
            group_name = message_data.get('group_name')
            message_text = message_data.get('message_text')
            user_id = message_data.get('user_id')

            if not all([token_address, group_name, message_text]):
                logger.warning("Missing required message data")
                return False

            # Get SpyDefi signals for additional context
            try:
                signals = self.spydefi.get_social_signals(token_address, chain)
                sentiment_score = signals.get('aggregated_sentiment', 0.5) if signals else 0.5
            except Exception as e:
                logger.error(f"Error getting SpyDefi signals: {str(e)}")
                sentiment_score = 0.5
                self._circuit_breaker['failures'] += 1
                if self._circuit_breaker['failures'] >= self._circuit_breaker['max_failures']:
                    self._circuit_breaker['is_open'] = True

            # Store message with enhanced metadata
            message = TelegramMessages(
                token_address=token_address,
                chain=chain,
                group_name=group_name,
                message_text=message_text,
                timestamp=datetime.utcnow(),
                sentiment=sentiment_score
            )

            # Check if message is from a KOL
            kol = KOLMetrics.query.filter_by(user_id=user_id).first()
            if kol:
                message.kol_id = kol.user_id
                message.user_influence_score = kol.influence_score
                message.kol_volume = kol.average_volume
                message.kol_success_rate = kol.success_rate
                message.kol_avg_gain = kol.avg_gain_multiple
                message.kol_total_2x_calls = kol.total_2x_calls

            # Add to batch for processing
            self.message_batch.append(message)

            # Process batch if size threshold reached
            if len(self.message_batch) >= self.batch_size:
                self._process_message_batch()

            return True

        except Exception as e:
            logger.error(f"Error processing Telegram message: {str(e)}")
            self._circuit_breaker['failures'] += 1
            return False

    def _process_message_batch(self):
        """Process batch of messages and update sentiment with SpyDefi integration"""
        try:
            if not self.message_batch:
                return

            # Group messages by token and chain
            message_groups = {}
            for msg in self.message_batch:
                key = (msg.token_address, msg.chain, msg.group_name)
                if key not in message_groups:
                    message_groups[key] = []
                message_groups[key].append(msg)

            # Update sentiment for each group
            for (token_address, chain, group_name), messages in message_groups.items():
                sentiment = TelegramSentiment.query.filter_by(
                    token_address=token_address,
                    chain=chain,
                    group_name=group_name
                ).first()

                if not sentiment:
                    sentiment = TelegramSentiment(
                        token_address=token_address,
                        chain=chain,
                        group_name=group_name
                    )

                # Update message counts
                sentiment.message_count += len(messages)

                # Calculate KOL-weighted sentiment
                total_score = 0
                total_weight = 0

                for msg in messages:
                    weight = msg.user_influence_score if msg.user_influence_score else 1.0
                    total_score += msg.sentiment * weight if msg.sentiment else 0
                    total_weight += weight

                if total_weight > 0:
                    sentiment.kol_weighted_score = total_score / total_weight

                # Get SpyDefi rating
                try:
                    if self._check_circuit_breaker():
                        spydefi_signals = self.spydefi.get_social_signals(token_address, chain)
                        if spydefi_signals:
                            sentiment.spydefi_rating = spydefi_signals.get('signal_strength', 'neutral')
                except Exception as e:
                    logger.error(f"Error getting SpyDefi signals: {str(e)}")
                    self._circuit_breaker['failures'] += 1

                sentiment.last_updated = datetime.utcnow()
                db.session.add(sentiment)

            # Save messages
            db.session.add_all(self.message_batch)
            db.session.commit()

            # Clear batch
            self.message_batch = []

        except Exception as e:
            logger.error(f"Error processing message batch: {str(e)}")
            db.session.rollback()

    def get_token_sentiment(self, token_address, chain):
        """Get aggregated sentiment data for a token with SpyDefi insights"""
        try:
            if not self._check_rate_limit():
                logger.warning("Rate limit exceeded")
                return {'error': 'Rate limit exceeded'}, 429

            sentiments = TelegramSentiment.query.filter_by(
                token_address=token_address,
                chain=chain
            ).all()

            if not sentiments:
                # Get SpyDefi signals even if no Telegram data
                try:
                    if self._check_circuit_breaker():
                        spydefi_data = self.spydefi.get_social_signals(token_address, chain)
                        if spydefi_data:
                            return {
                                'token_address': token_address,
                                'chain': chain,
                                'sentiment_score': spydefi_data.get('aggregated_sentiment', 0),
                                'message_count': 0,
                                'spydefi_metrics': spydefi_data,
                                'groups': []
                            }
                except Exception as e:
                    logger.error(f"Error getting SpyDefi data: {str(e)}")
                    self._circuit_breaker['failures'] += 1

                return {
                    'token_address': token_address,
                    'chain': chain,
                    'sentiment_score': 0,
                    'message_count': 0,
                    'groups': []
                }

            total_messages = sum(s.message_count for s in sentiments)
            weighted_score = sum(s.kol_weighted_score * s.message_count for s in sentiments if s.kol_weighted_score)
            avg_sentiment = weighted_score / total_messages if total_messages > 0 else 0

            # Add SpyDefi insights
            spydefi_data = None
            try:
                if self._check_circuit_breaker():
                    spydefi_data = self.spydefi.get_social_signals(token_address, chain)
            except Exception as e:
                logger.error(f"Error getting SpyDefi signals: {str(e)}")
                self._circuit_breaker['failures'] += 1

            return {
                'token_address': token_address,
                'chain': chain,
                'sentiment_score': avg_sentiment,
                'message_count': total_messages,
                'spydefi_metrics': spydefi_data if spydefi_data else {},
                'groups': [s.to_dict() for s in sentiments]
            }

        except Exception as e:
            logger.error(f"Error getting token sentiment: {str(e)}")
            return None