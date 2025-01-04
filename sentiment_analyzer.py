"""
Sentiment Analyzer Module
Responsible for analyzing social media sentiment and market sentiment tracking
"""

import logging
import tweepy
import requests
from textblob import TextBlob
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    def __init__(self):
        # Initialize with default thresholds
        self.sentiment_thresholds = {
            'positive': 0.3,
            'negative': -0.3,
            'volume_threshold': 100  # Minimum message volume for reliable sentiment
        }

        # Cache to store sentiment results
        self.sentiment_cache = {}
        self.cache_duration = timedelta(minutes=15)

    def analyze_token(self, token_address: str) -> Dict[str, Any]:
        """
        Analyze sentiment for a specific token across multiple platforms
        Returns a normalized sentiment score and confidence level
        """
        try:
            # Check cache first
            cached_result = self._get_cached_sentiment(token_address)
            if cached_result:
                return cached_result

            # Combine different sentiment sources with weights
            twitter_sentiment = self._analyze_twitter(token_address)
            telegram_sentiment = self._analyze_telegram(token_address)

            # Calculate weighted sentiment score
            weighted_score = (
                twitter_sentiment.get('score', 0) * 0.6 +
                telegram_sentiment.get('score', 0) * 0.4
            )

            # Calculate confidence based on message volume
            total_volume = (
                twitter_sentiment.get('volume', 0) +
                telegram_sentiment.get('volume', 0)
            )

            confidence = min(total_volume / self.sentiment_thresholds['volume_threshold'], 1.0)

            result = {
                'token_address': token_address,
                'overall_score': weighted_score,
                'confidence': confidence,
                'timestamp': datetime.utcnow().isoformat(),
                'sources': {
                    'twitter': twitter_sentiment,
                    'telegram': telegram_sentiment
                }
            }

            # Cache the result
            self._cache_sentiment(token_address, result)

            return result

        except Exception as e:
            logger.error(f"Error analyzing sentiment for token {token_address}: {str(e)}")
            return {
                'token_address': token_address,
                'overall_score': 0,
                'confidence': 0,
                'error': str(e)
            }

    def _analyze_twitter(self, token_address: str) -> Dict[str, Any]:
        """Analyze Twitter sentiment for a token"""
        try:
            # Get recent tweets mentioning the token
            tweets = self._get_token_tweets(token_address)
            if not tweets:
                return {'score': 0, 'volume': 0}

            sentiment_scores = []
            for tweet in tweets:
                analysis = TextBlob(tweet.text)
                sentiment_scores.append(analysis.sentiment.polarity)

            return {
                'score': sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0,
                'volume': len(tweets)
            }

        except Exception as e:
            logger.error(f"Twitter analysis error for {token_address}: {str(e)}")
            return {'score': 0, 'volume': 0, 'error': str(e)}

    def _analyze_telegram(self, token_address: str) -> Dict[str, Any]:
        """Analyze Telegram sentiment for a token"""
        try:
            # Get messages from relevant Telegram channels
            messages = self._get_telegram_messages(token_address)
            if not messages:
                return {'score': 0, 'volume': 0}

            sentiment_scores = []
            for message in messages:
                analysis = TextBlob(message)
                sentiment_scores.append(analysis.sentiment.polarity)

            return {
                'score': sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0,
                'volume': len(messages)
            }

        except Exception as e:
            logger.error(f"Telegram analysis error for {token_address}: {str(e)}")
            return {'score': 0, 'volume': 0, 'error': str(e)}

    def _get_token_tweets(self, token_address: str) -> list:
        """Fetch recent tweets about the token"""
        try:
            # Implement actual Twitter API call here
            # For MVP, return empty list to avoid rate limits
            return []
        except Exception as e:
            logger.error(f"Error fetching tweets: {str(e)}")
            return []

    def _get_telegram_messages(self, token_address: str) -> list:
        """Fetch recent Telegram messages about the token"""
        try:
            # Implement actual Telegram API call here
            # For MVP, return empty list
            return []
        except Exception as e:
            logger.error(f"Error fetching Telegram messages: {str(e)}")
            return []

    def _get_cached_sentiment(self, token_address: str) -> Optional[Dict[str, Any]]:
        """Get cached sentiment if available and not expired"""
        if token_address in self.sentiment_cache:
            cached_data = self.sentiment_cache[token_address]
            if datetime.utcnow() - cached_data['timestamp'] < self.cache_duration:
                return cached_data['data']
        return None

    def _cache_sentiment(self, token_address: str, sentiment_data: Dict[str, Any]):
        """Cache sentiment analysis results"""
        self.sentiment_cache[token_address] = {
            'timestamp': datetime.utcnow(),
            'data': sentiment_data
        }