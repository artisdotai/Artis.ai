"""
Portfolio optimization service for Eliza Framework
"""
import logging
from typing import Dict, Any, List
from datetime import datetime
import numpy as np
from ..database import db
from ..models.analysis import TokenAnalysis

logger = logging.getLogger(__name__)

class PortfolioOptimizer:
    """Portfolio optimization service implementation"""
    
    def optimize_allocation(self, positions: List[Dict[str, Any]], risk_parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize portfolio allocation based on positions and risk parameters"""
        try:
            # Validate inputs
            if not positions:
                return {
                    'status': 'error',
                    'message': 'No positions provided for optimization',
                    'timestamp': datetime.utcnow().isoformat()
                }
                
            # Get token analyses for all positions
            token_analyses = {}
            for position in positions:
                analysis = TokenAnalysis.query.filter_by(
                    token_address=position['token_address']
                ).order_by(TokenAnalysis.created_at.desc()).first()
                
                if analysis:
                    token_analyses[position['token_address']] = analysis
            
            # Calculate risk metrics
            risk_metrics = self._calculate_risk_metrics(positions, token_analyses)
            
            # Apply risk constraints
            max_allocation = risk_parameters.get('max_allocation', 0.25)
            min_allocation = risk_parameters.get('min_allocation', 0.05)
            risk_tolerance = risk_parameters.get('risk_tolerance', 0.5)
            
            # Calculate optimal allocations
            allocations = self._optimize_weights(
                positions,
                risk_metrics,
                max_allocation,
                min_allocation,
                risk_tolerance
            )
            
            return {
                'status': 'success',
                'optimized_portfolio': {
                    'allocations': allocations,
                    'risk_metrics': risk_metrics,
                    'parameters_used': {
                        'max_allocation': max_allocation,
                        'min_allocation': min_allocation,
                        'risk_tolerance': risk_tolerance
                    }
                },
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Portfolio optimization error: {str(e)}")
            return {
                'status': 'error',
                'message': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    def _calculate_risk_metrics(self, positions: List[Dict[str, Any]], 
                              token_analyses: Dict[str, TokenAnalysis]) -> Dict[str, Any]:
        """Calculate risk metrics for positions"""
        risk_metrics = {}
        
        for position in positions:
            token_address = position['token_address']
            analysis = token_analyses.get(token_address)
            
            if analysis:
                risk_metrics[token_address] = {
                    'volatility': analysis.risk_score / 100 if analysis.risk_score else 0.5,
                    'liquidity_score': min(1.0, analysis.liquidity / 1000000) if analysis.liquidity else 0.3,
                    'sentiment_score': analysis.sentiment_score if analysis.sentiment_score else 0.5,
                    'overall_risk': 0.0  # Will be calculated below
                }
                
                # Calculate overall risk score (weighted average of components)
                metrics = risk_metrics[token_address]
                metrics['overall_risk'] = (
                    0.4 * metrics['volatility'] +
                    0.3 * (1 - metrics['liquidity_score']) +
                    0.3 * (1 - metrics['sentiment_score'])
                )
            else:
                # Use default risk metrics if no analysis available
                risk_metrics[token_address] = {
                    'volatility': 0.5,
                    'liquidity_score': 0.3,
                    'sentiment_score': 0.5,
                    'overall_risk': 0.5
                }
                
        return risk_metrics
    
    def _optimize_weights(self, positions: List[Dict[str, Any]], 
                         risk_metrics: Dict[str, Any],
                         max_allocation: float,
                         min_allocation: float,
                         risk_tolerance: float) -> List[Dict[str, Any]]:
        """Calculate optimal position weights"""
        # Initialize allocations
        n_positions = len(positions)
        base_allocation = 1.0 / n_positions
        
        # Calculate risk-adjusted allocations
        allocations = []
        remaining_allocation = 1.0
        
        for position in positions:
            token_address = position['token_address']
            risk_score = risk_metrics[token_address]['overall_risk']
            
            # Adjust allocation based on risk score and tolerance
            risk_factor = 1 - (risk_score * (1 - risk_tolerance))
            allocation = base_allocation * risk_factor
            
            # Apply constraints
            allocation = min(max(allocation, min_allocation), max_allocation)
            
            allocations.append({
                'token_address': token_address,
                'allocation': allocation,
                'risk_score': risk_score,
                'risk_adjusted_factor': risk_factor
            })
            
            remaining_allocation -= allocation
            
        # Normalize allocations if needed
        if remaining_allocation > 0:
            total_weight = sum(a['risk_adjusted_factor'] for a in allocations)
            for allocation in allocations:
                extra = (remaining_allocation * allocation['risk_adjusted_factor'] / total_weight)
                allocation['allocation'] = min(allocation['allocation'] + extra, max_allocation)
                
        return allocations
