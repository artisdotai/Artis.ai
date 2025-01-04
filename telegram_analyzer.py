import logging
from datetime import datetime, timedelta
import json
from models import db, TelegramSentiment, TelegramMessages, KOLMetrics
from textblob import TextBlob
import requests

logger = logging.getLogger(__name__)

class TelegramAnalyzer:
    def __init__(self):
        self.spydefi_group = "SpydefiSignals"  # Main Spydefi signals channel
        self.watched_groups = [
            self.spydefi_group,
            "BSCGems",
            "UniswapGems", 
            "dexgemschat",
            "CryptoVIPsignals"
        ]
        # Enhanced verification parameters 
        self.min_volume_threshold = 500000  # Increased to $500k minimum volume
        self.min_success_rate = 0.80  # Increased to 80% minimum success rate
        self.min_gain_multiple = 2.5  # Increased minimum gain to 2.5x
        self.min_successful_2x_calls = 5  # Increased to minimum 5 successful 2x calls
        self.max_age_hours = 12  # Only analyze launches within 12 hours
        self.min_liquidity = 100000  # Minimum $100k liquidity
        self.kol_cache = {}
        self.kol_cache_duration = timedelta(hours=1)

    def _is_trusted_kol(self, metrics):
        """Enhanced verification check for KOLs with proven 2.5x gains"""
        if not metrics:
            return False

        # More stringent requirements for trusted KOLs
        return (
            metrics.get('avg_volume', 0) >= self.min_volume_threshold and
            metrics.get('success_rate', 0) >= self.min_success_rate and
            metrics.get('avg_gain_multiple', 0) >= self.min_gain_multiple and
            metrics.get('total_2x_calls', 0) >= self.min_successful_2x_calls and
            metrics.get('best_gain_multiple', 0) >= 3.0  # Must have at least one 3x gain
        )

    def _is_new_launch(self, token_data):
        """Stricter verification for new launches"""
        if not token_data:
            return False

        # Check if token was created within max age hours
        creation_time = token_data.get('creation_time')
        if not creation_time:
            return False

        try:
            creation_dt = datetime.fromtimestamp(creation_time)
            time_since_launch = datetime.utcnow() - creation_dt

            # Only consider very early launches
            if time_since_launch > timedelta(hours=self.max_age_hours):
                return False

            # Enhanced launch criteria
            return (
                token_data.get('holder_count', 0) < 100 and  # Early holder count
                not token_data.get('is_verified', False) and  # Not yet verified
                float(token_data.get('total_supply', 0)) > 0 and  # Valid supply
                float(token_data.get('liquidity', 0)) >= self.min_liquidity  # Minimum liquidity
            )
        except Exception as e:
            logger.error(f"Error checking launch time: {str(e)}")
            return False

    async def analyze_spydefi_signals(self, token_address, chain):
        """Focus specifically on Spydefi signals with enhanced verification"""
        try:
            # Get messages from Spydefi channel about the token
            messages = await self._fetch_group_messages(self.spydefi_group, token_address)
            if not messages:
                return None

            # Get KOL metrics for message authors
            kol_metrics = await self._get_kol_metrics([msg['user_id'] for msg in messages])

            # Filter messages from trusted KOLs with proven 2x history
            trusted_messages = [
                msg for msg in messages 
                if self._is_trusted_kol(kol_metrics.get(msg['user_id']))
            ]

            if not trusted_messages:
                logger.info(f"No trusted KOL signals found for {token_address}")
                return None

            # Get token data to check if it's a new launch
            token_data = await self._get_token_data(token_address, chain)
            if not self._is_new_launch(token_data):
                logger.info(f"Skipping {token_address} - not a new launch")
                return None

            # Analyze sentiment with KOL weighting
            sentiment_data = self._analyze_messages(trusted_messages, kol_metrics)

            # Store results
            await self._store_sentiment_data(
                token_address=token_address,
                chain=chain,
                group_name=self.spydefi_group,
                messages=trusted_messages,
                sentiment_data=sentiment_data,
                spydefi_rating='VERIFIED_KOL',  # Special rating for verified KOLs
                kol_metrics=kol_metrics,
                is_new_launch=True
            )

            return sentiment_data

        except Exception as e:
            logger.error(f"Error analyzing Spydefi signals: {str(e)}")
            return None

    async def analyze_group_sentiment(self, token_address, chain):
        """Analyze sentiment with enhanced KOL tracking for new launches"""
        try:
            for group in self.watched_groups:
                # Get messages from group about the token
                messages = await self._fetch_group_messages(group, token_address)
                if not messages:
                    continue

                # Get KOL metrics for message authors
                kol_metrics = await self._get_kol_metrics([msg['user_id'] for msg in messages])

                # Filter messages from trusted KOLs with 2x gain history
                trusted_messages = [
                    msg for msg in messages 
                    if self._is_trusted_kol(kol_metrics.get(msg['user_id']))
                ]

                # Only proceed if we have messages from trusted KOLs
                if not trusted_messages:
                    continue

                # Get token data to check if it's a new launch
                token_data = await self._get_token_data(token_address, chain)

                # Only analyze if it's a new launch
                if not self._is_new_launch(token_data):
                    logger.info(f"Skipping {token_address} - not a new launch")
                    continue

                # Analyze sentiment with KOL weighting
                sentiment_data = self._analyze_messages(trusted_messages, kol_metrics)

                # Get Spydefi rating
                spydefi_rating = await self._get_spydefi_rating(token_address, chain)

                # Store results with KOL attribution
                await self._store_sentiment_data(
                    token_address=token_address,
                    chain=chain,
                    group_name=group,
                    messages=trusted_messages,
                    sentiment_data=sentiment_data,
                    spydefi_rating=spydefi_rating,
                    kol_metrics=kol_metrics,
                    is_new_launch=True
                )

            return True

        except Exception as e:
            logger.error(f"Error analyzing group sentiment: {str(e)}")
            return False

    async def _get_token_data(self, token_address, chain):
        """Get token data including creation time"""
        try:
            # Implementation would connect to blockchain for token data
            # For testing, return mock data
            return {
                'creation_time': int((datetime.utcnow() - timedelta(hours=2)).timestamp()),
                'is_verified': False,
                'total_supply': 1000000000,
                'holder_count': 10,
                'liquidity': 150000
            }
        except Exception as e:
            logger.error(f"Error getting token data: {str(e)}")
            return None

    async def _get_kol_metrics(self, user_ids):
        """Get cached or fetch new KOL metrics"""
        metrics = {}
        uncached_ids = []
        now = datetime.utcnow()

        # Check cache first
        for user_id in user_ids:
            if user_id in self.kol_cache:
                cached_data, cache_time = self.kol_cache[user_id]
                if now - cache_time < self.kol_cache_duration:
                    metrics[user_id] = cached_data
                    continue
            uncached_ids.append(user_id)

        if uncached_ids:
            # Fetch from database
            db_metrics = KOLMetrics.query.filter(
                KOLMetrics.user_id.in_(uncached_ids)
            ).all()

            for metric in db_metrics:
                metrics[metric.user_id] = {
                    'avg_volume': metric.average_volume,
                    'success_rate': metric.success_rate,
                    'total_calls': metric.total_calls,
                    'influence_score': metric.influence_score,
                    'last_updated': metric.last_updated,
                    'avg_gain_multiple': metric.avg_gain_multiple,
                    'total_2x_calls': metric.total_2x_calls,
                    'best_gain_multiple': metric.best_gain_multiple
                }
                # Update cache
                self.kol_cache[metric.user_id] = (metrics[metric.user_id], now)

        return metrics

    async def _store_sentiment_data(self, token_address, chain, group_name, messages, 
                                  sentiment_data, spydefi_rating, kol_metrics, is_new_launch):
        """Store sentiment analysis results with enhanced KOL attribution"""
        try:
            # Store overall sentiment
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

            sentiment.message_count = sentiment_data['message_count']
            sentiment.positive_count = sentiment_data['positive_count']
            sentiment.negative_count = sentiment_data['negative_count']
            sentiment.neutral_count = sentiment_data['neutral_count']
            sentiment.sentiment_score = sentiment_data['sentiment_score']
            sentiment.spydefi_rating = spydefi_rating
            sentiment.top_keywords = json.dumps(sentiment_data['top_keywords'])
            sentiment.kol_weighted_score = sentiment_data.get('kol_weighted_score', 0)
            sentiment.is_new_launch = is_new_launch
            sentiment.last_updated = datetime.utcnow()

            db.session.add(sentiment)

            # Store individual messages with enhanced KOL data
            for msg in messages:
                kol_data = kol_metrics.get(msg['user_id'], {})
                message = TelegramMessages(
                    token_address=token_address,
                    chain=chain,
                    group_name=group_name,
                    message_text=msg['text'],
                    sentiment=TextBlob(msg['text']).sentiment.polarity,
                    timestamp=msg['timestamp'],
                    user_influence_score=msg['user_influence'],
                    kol_id=msg['user_id'],
                    kol_volume=kol_data.get('avg_volume', 0),
                    kol_success_rate=kol_data.get('success_rate', 0),
                    kol_avg_gain=kol_data.get('avg_gain_multiple', 0),
                    kol_total_2x_calls=kol_data.get('total_2x_calls', 0),
                    is_new_launch=is_new_launch
                )
                db.session.add(message)

            db.session.commit()

        except Exception as e:
            logger.error(f"Error storing sentiment data: {str(e)}")
            db.session.rollback()

    async def _fetch_group_messages(self, group_name, token_address):
        """Fetch messages with enhanced KOL tracking"""
        try:
            # In production, this would use Telegram API
            # For testing, return simulated messages with KOL data
            return [
                {
                    'text': f"Great potential for {token_address}! Strong buy signal",
                    'timestamp': datetime.utcnow(),
                    'user_id': 'kol_1',
                    'user_influence': 1.2,
                    'historical_volume': 500000
                },
                {
                    'text': f"New launch alert! {token_address} launching soon",
                    'timestamp': datetime.utcnow(),
                    'user_id': 'kol_2',
                    'user_influence': 1.5,
                    'historical_volume': 750000
                }
            ]
        except Exception as e:
            logger.error(f"Error fetching messages from {group_name}: {str(e)}")
            return []

    def _analyze_messages(self, messages, kol_metrics):
        """Analyze sentiment with KOL weighting"""
        try:
            positive_count = 0
            negative_count = 0
            neutral_count = 0
            total_score = 0
            keywords = {}
            volume_weighted_sentiment = 0
            total_volume_weight = 0

            for message in messages:
                # Get KOL metrics
                kol_data = kol_metrics.get(message['user_id'], {})
                kol_influence = kol_data.get('influence_score', 1.0)

                # Analyze sentiment
                blob = TextBlob(message['text'])
                sentiment = blob.sentiment.polarity

                # Weight sentiment by KOL influence and historical volume
                volume_weight = kol_data.get('avg_volume', 0) / self.min_volume_threshold
                weighted_sentiment = sentiment * kol_influence * volume_weight

                volume_weighted_sentiment += weighted_sentiment
                total_volume_weight += volume_weight

                # Count sentiments
                if sentiment > 0.1:
                    positive_count += 1
                elif sentiment < -0.1:
                    negative_count += 1
                else:
                    neutral_count += 1

                # Extract keywords with KOL weighting
                words = message['text'].lower().split()
                for word in words:
                    if len(word) > 3:  # Skip short words
                        keywords[word] = keywords.get(word, 0) + kol_influence

            # Calculate final sentiment score
            message_count = len(messages)
            if message_count > 0 and total_volume_weight > 0:
                final_sentiment = volume_weighted_sentiment / total_volume_weight
            else:
                final_sentiment = 0

            # Get top keywords
            sorted_keywords = sorted(keywords.items(), key=lambda x: x[1], reverse=True)
            top_keywords = [word for word, count in sorted_keywords[:5]]

            return {
                'message_count': message_count,
                'positive_count': positive_count,
                'negative_count': negative_count,
                'neutral_count': neutral_count,
                'sentiment_score': final_sentiment,
                'top_keywords': top_keywords,
                'kol_weighted_score': volume_weighted_sentiment / max(1, total_volume_weight)
            }

        except Exception as e:
            logger.error(f"Error analyzing messages: {str(e)}")
            return None

    async def _get_spydefi_rating(self, token_address, chain):
        """Get token rating from Spydefi signals"""
        try:
            # Get messages from trusted KOLs about this token
            messages = await self._fetch_group_messages("SpydefiSignals", token_address)
            if not messages:
                return None

            # Get KOL metrics
            kol_metrics = await self._get_kol_metrics([msg['user_id'] for msg in messages])

            # Filter for trusted KOLs
            trusted_messages = [
                msg for msg in messages 
                if self._is_trusted_kol(kol_metrics.get(msg['user_id']))
            ]

            if not trusted_messages:
                return None

            # Calculate weighted rating based on KOL influence
            total_weight = 0
            weighted_rating = 0

            for msg in trusted_messages:
                kol_data = kol_metrics.get(msg['user_id'], {})
                weight = kol_data.get('influence_score', 1.0)

                # Simple rating extraction (implement more sophisticated logic in production)
                if 'safe' in msg['text'].lower():
                    rating = 1.0
                elif 'medium' in msg['text'].lower():
                    rating = 0.5
                else:
                    rating = 0.2

                weighted_rating += rating * weight
                total_weight += weight

            final_rating = weighted_rating / total_weight if total_weight > 0 else None

            # Convert to risk level
            if final_rating is None:
                return None
            elif final_rating >= 0.8:
                return 'SAFE'
            elif final_rating >= 0.4:
                return 'MEDIUM'
            else:
                return 'HIGH_RISK'

        except Exception as e:
            logger.error(f"Error getting Spydefi rating: {str(e)}")
            return None