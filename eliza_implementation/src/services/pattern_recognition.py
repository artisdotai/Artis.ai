"""Pattern Recognition Service for Eliza Framework"""
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
from functools import lru_cache

logger = logging.getLogger(__name__)

class PatternRecognizer:
    """Trading Pattern Recognition Service"""

    def __init__(self):
        """Initialize pattern recognition service"""
        self.patterns = {
            'bullish_flag': self._check_bullish_flag,
            'double_bottom': self._check_double_bottom,
            'head_shoulders': self._check_head_shoulders,
            'triangle': self._check_triangle
        }
        self._cache = {}
        self._cache_timeout = 300  # 5 minutes

    @lru_cache(maxsize=100)
    def _calculate_trend_strength(self, prices: tuple) -> float:
        """Calculate trend strength from price data"""
        if not prices or len(prices) < 2:
            return 0.0

        changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        positive_changes = sum(1 for c in changes if c > 0)
        return positive_changes / len(changes)

    def _check_bullish_flag(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Check for bullish flag pattern"""
        try:
            prices = data.get('prices', [])
            volumes = data.get('volumes', [])

            if not prices or len(prices) < 10:
                return {
                    'detected': False,
                    'confidence': 0.0,
                    'metrics': {}
                }

            # Calculate metrics
            trend_strength = self._calculate_trend_strength(tuple(prices))
            volume_increase = (volumes[-1] / volumes[0]) if volumes else 1.0
            price_change = ((prices[-1] - prices[0]) / prices[0]) * 100

            # Pattern detection logic
            is_consolidating = max(prices[-5:]) - min(prices[-5:]) < (prices[-1] * 0.03)
            has_volume_confirmation = volume_increase > 1.2
            has_higher_lows = all(prices[i] >= prices[i-1] for i in range(1, len(prices)))

            confidence = (
                (0.4 if is_consolidating else 0.0) +
                (0.3 if has_volume_confirmation else 0.0) +
                (0.3 if has_higher_lows else 0.0)
            )

            return {
                'detected': confidence > 0.5,
                'confidence': confidence,
                'metrics': {
                    'price_change': price_change,
                    'volume_change': volume_increase * 100 - 100,
                    'trend_strength': trend_strength
                }
            }

        except Exception as e:
            logger.error(f"Error analyzing bullish flag pattern: {str(e)}")
            return {
                'detected': False,
                'confidence': 0.0,
                'metrics': {}
            }

    def _check_double_bottom(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Check for double bottom pattern"""
        try:
            prices = data.get('prices', [])
            volumes = data.get('volumes', [])

            if not prices or len(prices) < 20:
                return {
                    'detected': False,
                    'confidence': 0.0,
                    'metrics': {}
                }

            # Calculate metrics
            lowest_points = sorted(prices)[:2]
            price_change = ((prices[-1] - min(prices)) / min(prices)) * 100
            volume_trend = (volumes[-1] / volumes[0]) if volumes else 1.0

            # Pattern detection logic
            bottoms_similar = abs(lowest_points[0] - lowest_points[1]) < (lowest_points[0] * 0.02)
            recovery_confirmed = prices[-1] > (max(prices) * 0.95)
            volume_confirmation = volume_trend > 1.15

            confidence = (
                (0.4 if bottoms_similar else 0.0) +
                (0.4 if recovery_confirmed else 0.0) +
                (0.2 if volume_confirmation else 0.0)
            )

            return {
                'detected': confidence > 0.6,
                'confidence': confidence,
                'metrics': {
                    'price_change': price_change,
                    'volume_trend': volume_trend * 100 - 100,
                    'bottom_difference': abs(lowest_points[0] - lowest_points[1])
                }
            }

        except Exception as e:
            logger.error(f"Error analyzing double bottom pattern: {str(e)}")
            return {
                'detected': False,
                'confidence': 0.0,
                'metrics': {}
            }

    def _check_head_shoulders(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Check for head and shoulders pattern"""
        try:
            prices = data.get('prices', [])
            volumes = data.get('volumes', [])

            if not prices or len(prices) < 20:
                return {
                    'detected': False,
                    'confidence': 0.0,
                    'metrics': {}
                }

            # Calculate metrics
            price_trend = ((prices[-1] - prices[0]) / prices[0]) * 100
            volume_trend = (volumes[-1] / volumes[0]) if volumes else 1.0

            # Find peaks and troughs
            peaks = []
            troughs = []
            for i in range(1, len(prices) - 1):
                if prices[i] > prices[i-1] and prices[i] > prices[i+1]:
                    peaks.append((i, prices[i]))
                elif prices[i] < prices[i-1] and prices[i] < prices[i+1]:
                    troughs.append((i, prices[i]))

            if len(peaks) < 3 or len(troughs) < 2:
                return {
                    'detected': False,
                    'confidence': 0.0,
                    'metrics': {}
                }

            # Pattern detection logic
            head_idx = max(peaks, key=lambda x: x[1])[0]
            left_shoulder = None
            right_shoulder = None
            neckline = None

            # Find shoulders
            for i, p in peaks:
                if i < head_idx and (not left_shoulder or p[1] > left_shoulder[1]):
                    left_shoulder = (i, p)
                elif i > head_idx and (not right_shoulder or p[1] > right_shoulder[1]):
                    right_shoulder = (i, p)

            if not (left_shoulder and right_shoulder):
                return {
                    'detected': False,
                    'confidence': 0.0,
                    'metrics': {}
                }

            # Find neckline
            neckline_troughs = [t for t in troughs if left_shoulder[0] < t[0] < right_shoulder[0]]
            if neckline_troughs:
                neckline = sum(t[1] for t in neckline_troughs) / len(neckline_troughs)

            # Calculate confidence
            symmetry = 1 - abs((left_shoulder[1] - right_shoulder[1]) / max(left_shoulder[1], right_shoulder[1]))
            head_prominence = (peaks[head_idx][1] - neckline) / neckline if neckline else 0
            volume_confirmation = volume_trend > 1.1

            confidence = (
                (0.4 * symmetry) +
                (0.4 * min(1.0, head_prominence)) +
                (0.2 if volume_confirmation else 0.0)
            )

            return {
                'detected': confidence > 0.6,
                'confidence': confidence,
                'metrics': {
                    'price_trend': price_trend,
                    'volume_trend': volume_trend * 100 - 100,
                    'symmetry': symmetry,
                    'head_prominence': head_prominence
                }
            }

        except Exception as e:
            logger.error(f"Error analyzing head and shoulders pattern: {str(e)}")
            return {
                'detected': False,
                'confidence': 0.0,
                'metrics': {}
            }

    def _check_triangle(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Check for triangle pattern"""
        try:
            prices = data.get('prices', [])
            volumes = data.get('volumes', [])

            if not prices or len(prices) < 15:
                return {
                    'detected': False,
                    'confidence': 0.0,
                    'metrics': {}
                }

            # Calculate metrics
            price_trend = ((prices[-1] - prices[0]) / prices[0]) * 100
            volume_trend = (volumes[-1] / volumes[0]) if volumes else 1.0

            # Find highs and lows
            highs = []
            lows = []
            for i in range(1, len(prices) - 1):
                if prices[i] > prices[i-1] and prices[i] > prices[i+1]:
                    highs.append((i, prices[i]))
                elif prices[i] < prices[i-1] and prices[i] < prices[i+1]:
                    lows.append((i, prices[i]))

            if len(highs) < 2 or len(lows) < 2:
                return {
                    'detected': False,
                    'confidence': 0.0,
                    'metrics': {}
                }

            # Calculate trend lines
            upper_slope = (highs[-1][1] - highs[0][1]) / (highs[-1][0] - highs[0][0])
            lower_slope = (lows[-1][1] - lows[0][1]) / (lows[-1][0] - lows[0][0])

            # Pattern detection logic
            is_ascending = lower_slope > 0 and abs(upper_slope) < abs(lower_slope)
            is_descending = upper_slope < 0 and abs(upper_slope) > abs(lower_slope)
            is_symmetric = abs(abs(upper_slope) - abs(lower_slope)) < 0.1

            # Volume pattern verification
            volume_declining = all(volumes[i] <= volumes[i-1] for i in range(1, len(volumes)))

            # Calculate pattern confidence
            slope_convergence = 1 - abs(abs(upper_slope) - abs(lower_slope))
            price_range_shrinking = (highs[-1][1] - lows[-1][1]) < (highs[0][1] - lows[0][1])

            confidence = 0.0
            if is_ascending:
                confidence = (
                    (0.4 if slope_convergence > 0.7 else 0.2) +
                    (0.3 if price_range_shrinking else 0.0) +
                    (0.3 if volume_declining else 0.0)
                )
            elif is_descending:
                confidence = (
                    (0.4 if slope_convergence > 0.7 else 0.2) +
                    (0.3 if price_range_shrinking else 0.0) +
                    (0.3 if volume_declining else 0.0)
                )
            elif is_symmetric:
                confidence = (
                    (0.5 if slope_convergence > 0.8 else 0.3) +
                    (0.3 if price_range_shrinking else 0.0) +
                    (0.2 if volume_declining else 0.0)
                )

            pattern_type = (
                'ascending' if is_ascending else
                'descending' if is_descending else
                'symmetric' if is_symmetric else
                'unknown'
            )

            return {
                'detected': confidence > 0.5,
                'confidence': confidence,
                'metrics': {
                    'pattern_type': pattern_type,
                    'price_trend': price_trend,
                    'volume_trend': volume_trend * 100 - 100,
                    'slope_convergence': slope_convergence,
                    'upper_slope': upper_slope,
                    'lower_slope': lower_slope
                }
            }

        except Exception as e:
            logger.error(f"Error analyzing triangle pattern: {str(e)}")
            return {
                'detected': False,
                'confidence': 0.0,
                'metrics': {}
            }

    def _generate_summary(self, patterns: List[Dict[str, Any]], trend_strength: float) -> Dict[str, Any]:
        """Generate pattern analysis summary"""
        if not patterns:
            return {}

        # Find primary pattern with highest confidence
        primary_pattern = max(patterns, key=lambda x: x['confidence'])

        # Calculate confidence-weighted score
        weighted_score = sum(p['confidence'] * 10 for p in patterns) / len(patterns)

        return {
            'primary_pattern': primary_pattern['type'],
            'confidence_weighted': round(weighted_score, 2),
            'timeframe': primary_pattern['timeframe'],
            'recommendation': self._get_recommendation(primary_pattern, trend_strength)
        }

    def identify_patterns(self, token_data: Dict[str, Any], pattern_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """Identify trading patterns in token data"""
        try:
            # Validate input
            if not token_data or not isinstance(token_data, dict):
                raise ValueError("Invalid token data format")

            # Convert price list to tuple for caching
            if 'prices' in token_data:
                token_data['prices'] = tuple(token_data['prices'])

            # Use all patterns if none specified
            if pattern_types is None:
                pattern_types = list(self.patterns.keys())

            # Process patterns
            results = []
            for pattern_type in pattern_types:
                if pattern_type not in self.patterns:
                    logger.warning(f"Unsupported pattern type: {pattern_type}")
                    continue

                pattern_result = self.patterns[pattern_type](token_data)
                if pattern_result['detected']:
                    results.append({
                        'type': pattern_type,
                        'confidence': pattern_result['confidence'],
                        'timeframe': token_data.get('timeframe', '4h'),
                        'signals': self._get_pattern_signals(pattern_type, pattern_result),
                        'metrics': pattern_result['metrics']
                    })

            # Calculate overall metrics
            trend_strength = self._calculate_trend_strength(token_data.get('prices', ()))

            return {
                'status': 'success',
                'patterns': results,
                'metrics': {
                    'trend_strength': trend_strength,
                    'volatility': self._calculate_volatility(token_data.get('prices', ())),
                    'momentum': self._calculate_momentum(token_data),
                    'market_sentiment': self._determine_sentiment(results),
                    'risk_level': self._assess_risk(results, trend_strength)
                },
                'timestamp': datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Error identifying patterns: {str(e)}")
            return {
                'status': 'error',
                'message': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }

    def _get_pattern_signals(self, pattern_type: str, result: Dict[str, Any]) -> List[str]:
        """Get relevant signals for detected pattern"""
        signals = []
        metrics = result.get('metrics', {})

        if pattern_type == 'bullish_flag':
            if metrics.get('volume_change', 0) > 20:
                signals.append('volume_increase')
            if metrics.get('price_change', 0) > 0:
                signals.append('price_consolidation')
            if metrics.get('trend_strength', 0) > 0.6:
                signals.append('higher_lows')

        elif pattern_type == 'double_bottom':
            if metrics.get('volume_trend', 0) > 15:
                signals.append('volume_confirmation')
            if metrics.get('price_change', 0) > 0:
                signals.append('support_level_test')
            if metrics.get('bottom_difference', float('inf')) < 0.02:
                signals.append('momentum_shift')

        elif pattern_type == 'head_shoulders':
            if metrics.get('volume_trend', 0) > 15:
                signals.append('volume_confirmation')
            if metrics.get('price_trend', 0) < 0:
                signals.append('price_reversal')
            if metrics.get('symmetry', 0) > 0.8:
                signals.append('pattern_symmetry')

        elif pattern_type == 'triangle':
            if metrics.get('volume_trend', 0) < -10:
                signals.append('volume_decline')
            if metrics.get('slope_convergence', 0) > 0.7:
                signals.append('slope_convergence')
            if metrics.get('pattern_type') in ['ascending', 'descending']:
                signals.append('breakout_expected')


        return signals

    def _calculate_volatility(self, prices: tuple) -> float:
        """Calculate price volatility"""
        if not prices or len(prices) < 2:
            return 0.0

        changes = [abs(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
        return sum(changes) / len(changes)

    def _calculate_momentum(self, data: Dict[str, Any]) -> float:
        """Calculate price momentum"""
        prices = data.get('prices', ())
        if not prices or len(prices) < 2:
            return 0.0

        short_term = sum(prices[-5:]) / 5 if len(prices) >= 5 else prices[-1]
        long_term = sum(prices) / len(prices)

        return (short_term - long_term) / long_term

    def _determine_sentiment(self, patterns: List[Dict[str, Any]]) -> str:
        """Determine overall market sentiment"""
        if not patterns:
            return 'neutral'

        sentiment_score = sum(
            0.5 if p['type'] in ['bullish_flag', 'double_bottom'] else -0.5
            for p in patterns
        )

        if sentiment_score > 0.3:
            return 'bullish'
        elif sentiment_score < -0.3:
            return 'bearish'
        return 'neutral'

    def _assess_risk(self, patterns: List[Dict[str, Any]], trend_strength: float) -> str:
        """Assess overall risk level"""
        if not patterns:
            return 'high'

        risk_score = sum(p['confidence'] for p in patterns) / len(patterns)
        risk_score = (risk_score + trend_strength) / 2

        if risk_score > 0.7:
            return 'low'
        elif risk_score > 0.4:
            return 'medium'
        return 'high'

    def _get_recommendation(self, pattern: Dict[str, Any], trend_strength: float) -> str:
        """Generate trading recommendation based on pattern and trend"""
        if pattern['confidence'] < 0.6:
            return 'monitor_only'

        if pattern['type'] in ['bullish_flag', 'double_bottom'] and trend_strength > 0.6:
            return 'potential_entry'

        if pattern['type'] in ['head_shoulders', 'triangle']:
            return 'await_confirmation'

        return 'monitor_only'