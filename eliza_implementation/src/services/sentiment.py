"""
Sentiment analysis service for Eliza Framework
"""
import logging
from typing import Dict, Any, List
from datetime import datetime
from src.database import db
from src.models.analysis import SentimentAnalysis

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """Sentiment analysis service implementation"""

    def __init__(self):
        """Initialize sentiment analyzer"""
        self.sources = ['telegram', 'twitter', 'reddit']
        self.sentiment_thresholds = {
            'very_negative': -0.6,
            'negative': -0.2,
            'neutral': 0.2,
            'positive': 0.6,
            'very_positive': 1.0
        }

    def analyze_content(self, content: str, source: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze content sentiment"""
        try:
            # Initialize metadata if not provided
            metadata = metadata or {
                'content_hash': None,
                'sentiment_score': 0,
                'influence_score': 0,
                'mentions_count': 0,
                'engagement_rate': 0
            }

            # Create sentiment record
            analysis = SentimentAnalysis(
                source=source,
                content_hash=metadata.get('content_hash'),
                sentiment_score=metadata.get('sentiment_score', 0),
                influence_score=metadata.get('influence_score', 0),
                mentions_count=metadata.get('mentions_count', 0),
                engagement_rate=metadata.get('engagement_rate', 0)
            )

            db.session.add(analysis)
            db.session.commit()

            return {
                'id': analysis.id,
                'source': source,
                'sentiment': self._get_sentiment_label(analysis.sentiment_score),
                'influence': analysis.influence_score,
                'engagement': analysis.engagement_rate,
                'timestamp': analysis.created_at.isoformat()
            }

        except Exception as e:
            logger.error(f"Sentiment analysis error: {str(e)}")
            return {}

    def get_market_mood(self, timeframe: str = '24h') -> Dict[str, Any]:
        """Get overall market sentiment"""
        try:
            analyses = SentimentAnalysis.query.all()
            if not analyses:
                return {
                    'status': 'insufficient_data',
                    'mood': 'neutral',
                    'confidence': 0,
                    'sources': {}
                }

            # Calculate mood per source
            source_moods = {}
            for source in self.sources:
                source_analyses = [a for a in analyses if a.source == source]
                if source_analyses:
                    avg_score = sum(a.sentiment_score for a in source_analyses) / len(source_analyses)
                    source_moods[source] = {
                        'mood': self._get_sentiment_label(avg_score),
                        'score': avg_score,
                        'sample_size': len(source_analyses)
                    }

            # Calculate overall mood
            overall_score = sum(a.sentiment_score * a.influence_score for a in analyses) / \
                          sum(a.influence_score for a in analyses) if analyses else 0

            return {
                'status': 'success',
                'mood': self._get_sentiment_label(overall_score),
                'confidence': self._calculate_confidence(analyses),
                'sources': source_moods,
                'updated_at': datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Error getting market mood: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }

    def _get_sentiment_label(self, score: float) -> str:
        """Convert sentiment score to label"""
        if score <= self.sentiment_thresholds['very_negative']:
            return 'very_negative'
        elif score <= self.sentiment_thresholds['negative']:
            return 'negative'
        elif score <= self.sentiment_thresholds['neutral']:
            return 'neutral'
        elif score <= self.sentiment_thresholds['positive']:
            return 'positive'
        else:
            return 'very_positive'

    def _calculate_confidence(self, analyses: List[SentimentAnalysis]) -> float:
        """Calculate confidence score based on sample size and influence"""
        if not analyses:
            return 0.0

        total_influence = sum(a.influence_score for a in analyses)
        total_engagement = sum(a.engagement_rate for a in analyses)
        sample_factor = min(1.0, len(analyses) / 100)  # Normalize by expected sample size

        return min(1.0, (0.4 * sample_factor + 0.3 * total_influence + 0.3 * total_engagement))