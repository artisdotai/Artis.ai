"""
Historical analysis service for Eliza Framework
"""
import logging
from typing import Dict, Any
from datetime import datetime, timedelta
from ..database import db
from ..models.analysis import TokenAnalysis

logger = logging.getLogger(__name__)

class HistoricalAnalyzer:
    """Historical analysis service implementation"""
    
    def analyze_performance(self, token_address: str, time_range: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze historical performance for a token"""
        try:
            # Convert time range to timedelta
            time_map = {
                '24h': timedelta(hours=24),
                '7d': timedelta(days=7),
                '30d': timedelta(days=30),
                '90d': timedelta(days=90)
            }
            lookback = time_map.get(time_range, timedelta(days=7))
            
            # Get historical analysis records
            analyses = TokenAnalysis.query.filter(
                TokenAnalysis.token_address == token_address,
                TokenAnalysis.created_at >= datetime.utcnow() - lookback
            ).order_by(TokenAnalysis.created_at.desc()).all()
            
            if not analyses:
                return {
                    'status': 'no_data',
                    'message': f'No historical data available for {token_address}',
                    'timestamp': datetime.utcnow().isoformat()
                }
            
            # Calculate performance metrics
            price_changes = []
            volume_changes = []
            holder_counts = []
            sentiment_scores = []
            
            for i in range(1, len(analyses)):
                if analyses[i].price and analyses[i-1].price:
                    price_change = (analyses[i-1].price - analyses[i].price) / analyses[i].price
                    price_changes.append(price_change)
                
                if analyses[i].volume_24h and analyses[i-1].volume_24h:
                    volume_change = (analyses[i-1].volume_24h - analyses[i].volume_24h) / analyses[i].volume_24h
                    volume_changes.append(volume_change)
                
                if analyses[i].holder_count:
                    holder_counts.append(analyses[i].holder_count)
                    
                if analyses[i].sentiment_score:
                    sentiment_scores.append(analyses[i].sentiment_score)
            
            # Calculate averages and trends
            avg_price_change = sum(price_changes) / len(price_changes) if price_changes else 0
            avg_volume_change = sum(volume_changes) / len(volume_changes) if volume_changes else 0
            avg_holders = sum(holder_counts) / len(holder_counts) if holder_counts else 0
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
            
            latest = analyses[0]
            return {
                'status': 'success',
                'token_address': token_address,
                'time_range': time_range,
                'data_points': len(analyses),
                'current_metrics': {
                    'price': latest.price,
                    'volume_24h': latest.volume_24h,
                    'holder_count': latest.holder_count,
                    'sentiment_score': latest.sentiment_score
                },
                'performance': {
                    'price_change': avg_price_change,
                    'volume_change': avg_volume_change,
                    'avg_holders': avg_holders,
                    'avg_sentiment': avg_sentiment,
                    'trend': self._calculate_trend_strength(price_changes)
                },
                'analysis_period': {
                    'start': analyses[-1].created_at.isoformat(),
                    'end': latest.created_at.isoformat()
                },
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Historical analysis error for {token_address}: {str(e)}")
            return {
                'status': 'error',
                'message': str(e),
                'token_address': token_address,
                'timestamp': datetime.utcnow().isoformat()
            }
    
    def _calculate_trend_strength(self, price_changes: list) -> str:
        """Calculate trend strength based on price changes"""
        if not price_changes:
            return 'neutral'
            
        positive_changes = sum(1 for change in price_changes if change > 0)
        negative_changes = sum(1 for change in price_changes if change < 0)
        
        ratio = positive_changes / len(price_changes)
        
        if ratio >= 0.7:
            return 'strong_uptrend'
        elif ratio >= 0.6:
            return 'uptrend'
        elif ratio <= 0.3:
            return 'strong_downtrend'
        elif ratio <= 0.4:
            return 'downtrend'
        else:
            return 'neutral'
