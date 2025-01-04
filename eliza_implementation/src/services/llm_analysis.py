"""
LLM-based market analysis service for Eliza Framework
"""
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from ..database import db
from ..models.analysis import TokenAnalysis

logger = logging.getLogger(__name__)

class LLMAnalyzer:
    """LLM-powered market analysis service"""

    def __init__(self):
        """Initialize LLM analyzer with connector"""
        from .llm_connector import LLMConnector
        self.llm_connector = LLMConnector()

    def analyze_market_trends(self, chain: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze market trends using LLM insights"""
        try:
            # Prepare market data for analysis
            analysis_data = {
                'chain': chain,
                'metrics': metrics,
                'timestamp': datetime.utcnow().isoformat()
            }

            # Get LLM analysis with fallback support
            analysis = self.llm_connector.analyze_market_conditions(analysis_data)

            if not analysis:
                logger.warning(f"Failed to get LLM analysis for {chain}")
                return {
                    'status': 'error',
                    'message': 'Analysis unavailable',
                    'timestamp': datetime.utcnow().isoformat()
                }

            # Store analysis results
            try:
                # Try to find existing analysis
                token_analysis = TokenAnalysis.query.filter_by(
                    chain=chain
                ).first()

                if not token_analysis:
                    token_analysis = TokenAnalysis(
                        chain=chain,
                        price=metrics.get('price', 0),
                        market_cap=metrics.get('market_cap', 0),
                        liquidity=metrics.get('liquidity', 0),
                        volume_24h=metrics.get('volume_24h', 0),
                        holder_count=metrics.get('holder_count', 0),
                        risk_score=analysis.get('risk_level', 0.5),
                        sentiment_score=analysis.get('confidence_score', 0.5),
                        technical_rating=analysis.get('trend_analysis', 'neutral')[:250]  # Limit length
                    )
                else:
                    # Update existing analysis
                    token_analysis.price = metrics.get('price', 0)
                    token_analysis.market_cap = metrics.get('market_cap', 0)
                    token_analysis.liquidity = metrics.get('liquidity', 0)
                    token_analysis.volume_24h = metrics.get('volume_24h', 0)
                    token_analysis.holder_count = metrics.get('holder_count', 0)
                    token_analysis.risk_score = analysis.get('risk_level', 0.5)
                    token_analysis.sentiment_score = analysis.get('confidence_score', 0.5)
                    token_analysis.technical_rating = analysis.get('trend_analysis', 'neutral')[:250]

                db.session.add(token_analysis)
                db.session.commit()

            except Exception as db_error:
                logger.error(f"Database error saving analysis: {str(db_error)}")
                # Continue execution to return analysis even if storage fails

            return {
                'status': 'success',
                'analysis': {
                    'trend': analysis.get('trend_analysis'),
                    'risk_level': analysis.get('risk_level'),
                    'confidence': analysis.get('confidence_score'),
                    'action': analysis.get('recommended_action'),
                    'key_factors': analysis.get('key_factors', []),
                    'summary': analysis.get('trend_analysis')[:250]  # Include truncated summary
                },
                'metrics': {
                    'price': metrics.get('price'),
                    'market_cap': metrics.get('market_cap'),
                    'volume_24h': metrics.get('volume_24h')
                },
                'timestamp': datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Market analysis error: {str(e)}")
            return {
                'status': 'error',
                'message': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }

    def get_token_insights(self, token_address: str, chain: str) -> Optional[Dict[str, Any]]:
        """Get detailed token insights using LLM analysis"""
        try:
            # Get historical analysis
            analyses = TokenAnalysis.query.filter_by(
                token_address=token_address,
                chain=chain
            ).order_by(TokenAnalysis.created_at.desc()).limit(10).all()

            if not analyses:
                return None

            # Prepare historical data for LLM
            historical_data = [{
                'timestamp': a.created_at.isoformat(),
                'price': a.price,
                'volume': a.volume_24h,
                'liquidity': a.liquidity,
                'risk_score': a.risk_score,
                'sentiment_score': a.sentiment_score,
                'technical_rating': a.technical_rating
            } for a in analyses]

            # Get LLM insights
            llm_input = {
                'token_address': token_address,
                'chain': chain,
                'historical_data': historical_data,
                'request_type': 'token_insights'
            }

            insights = self.llm_connector.analyze_with_fallback(
                json.dumps(llm_input),
                analysis_type='token'
            )

            if not insights:
                return None

            return {
                'token_address': token_address,
                'chain': chain,
                'insights': insights,
                'updated_at': datetime.utcnow().isoformat(),
                'data_points': len(historical_data)
            }

        except Exception as e:
            logger.error(f"Token insights error: {str(e)}")
            return None

    def get_llm_status(self) -> Dict[str, Any]:
        """Get LLM service status"""
        return self.llm_connector.get_status()