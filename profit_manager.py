"""
Profit Manager Module
Handles profit allocation, buybacks, liquidity management and GPU upgrades
"""

from models import Trade, db
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class TradeResult:
    success: bool
    trade_details: Dict[str, Any]
    error: Optional[str] = None
    trade_type: str = 'BUY'

class ProfitManager:
    def __init__(self):
        # Profit allocation ratios
        self.allocation_ratios = {
            'buyback': 0.4,    # 40% for AI10X buybacks
            'liquidity': 0.3,  # 30% for liquidity additions
            'gpu': 0.2,        # 20% for GPU upgrades
            'trading': 0.1     # 10% for further trading
        }

        # Minimum thresholds for actions
        self.min_buyback_amount = 100    # Min USD for buyback
        self.min_liquidity_amount = 150  # Min USD for liquidity
        self.min_gpu_upgrade = 500       # Min USD for GPU upgrade

        # Accumulated amounts
        self.accumulated = {
            'buyback': 0.0,  # Changed to float
            'liquidity': 0.0,
            'gpu': 0.0,
            'trading': 0.0
        }

        # Performance tracking
        self.total_profits = 0.0
        self.total_buybacks = 0.0
        self.total_liquidity_added = 0.0
        self.total_gpu_upgrades = 0.0

    def process_trade(self, trade_result: TradeResult) -> Dict[str, Any]:
        """Process trade result and allocate profits"""
        if not trade_result.success:
            return {'success': False, 'error': 'Trade unsuccessful'}

        try:
            trade_details = trade_result.trade_details

            # Extract trade object if present
            trade = trade_details.get('trade')
            if not trade:
                logger.warning("No trade object found in trade details")
                return {'success': False, 'error': 'No trade object found'}

            # Calculate profit for completed trades
            if trade.trade_type == 'SELL':
                buy_trade = Trade.query.filter_by(
                    token_address=trade.token_address,
                    trade_type='BUY'
                ).order_by(Trade.timestamp.desc()).first()

                if buy_trade:
                    profit = (trade.price - buy_trade.price) * trade.amount
                    trade.profit = profit
                    self.total_profits += profit

                    # Allocate profits according to ratios
                    allocations = self._allocate_profits(profit)

                    # Process each allocation type
                    self._process_buyback_allocation(allocations['buyback'])
                    self._process_liquidity_allocation(allocations['liquidity'])
                    self._process_gpu_allocation(allocations['gpu'])
                    self._process_trading_allocation(allocations['trading'])

                    # Update trade record
                    trade.timestamp = datetime.utcnow()
                    db.session.commit()

                    logger.info(f"""
                        Trade processed:
                        Profit: ${profit:.2f}
                        Buyback allocated: ${allocations['buyback']:.2f}
                        Liquidity allocated: ${allocations['liquidity']:.2f}
                        GPU upgrade allocated: ${allocations['gpu']:.2f}
                        Trading capital: ${allocations['trading']:.2f}
                    """)

                    return {
                        'success': True,
                        'profit': profit,
                        'allocations': allocations
                    }

            return {'success': True, 'profit': 0}

        except Exception as e:
            logger.error(f"Profit management error: {str(e)}")
            db.session.rollback()
            return {'success': False, 'error': str(e)}

    def _allocate_profits(self, profit: float) -> Dict[str, float]:
        """Allocate profits according to defined ratios"""
        return {
            category: profit * ratio 
            for category, ratio in self.allocation_ratios.items()
        }

    def _process_buyback_allocation(self, amount: float):
        """Process buyback allocation"""
        try:
            self.accumulated['buyback'] += amount

            if self.accumulated['buyback'] >= self.min_buyback_amount:
                self._execute_buyback(self.accumulated['buyback'])
                self.accumulated['buyback'] = 0
            else:
                logger.info(
                    f"Accumulated buyback: ${self.accumulated['buyback']:.2f}, "
                    f"threshold: ${self.min_buyback_amount}"
                )

        except Exception as e:
            logger.error(f"Buyback allocation error: {str(e)}")

    def _process_liquidity_allocation(self, amount: float):
        """Process liquidity allocation"""
        try:
            self.accumulated['liquidity'] += amount

            if self.accumulated['liquidity'] >= self.min_liquidity_amount:
                self._add_liquidity(self.accumulated['liquidity'])
                self.accumulated['liquidity'] = 0
            else:
                logger.info(
                    f"Accumulated liquidity: ${self.accumulated['liquidity']:.2f}, "
                    f"threshold: ${self.min_liquidity_amount}"
                )

        except Exception as e:
            logger.error(f"Liquidity allocation error: {str(e)}")

    def _process_gpu_allocation(self, amount: float):
        """Process GPU upgrade allocation"""
        try:
            self.accumulated['gpu'] += amount

            if self.accumulated['gpu'] >= self.min_gpu_upgrade:
                self._execute_gpu_upgrade(self.accumulated['gpu'])
                self.accumulated['gpu'] = 0
            else:
                logger.info(
                    f"Accumulated GPU upgrade: ${self.accumulated['gpu']:.2f}, "
                    f"threshold: ${self.min_gpu_upgrade}"
                )

        except Exception as e:
            logger.error(f"GPU allocation error: {str(e)}")

    def _process_trading_allocation(self, amount: float):
        """Process trading capital allocation"""
        try:
            self.accumulated['trading'] += amount
            logger.info(f"Added ${amount:.2f} to trading capital pool")

            # Here we would update the trading capital pool
            # For now, we just log the allocation
            self._allocate_trading_capital(amount)

        except Exception as e:
            logger.error(f"Trading capital allocation error: {str(e)}")

    def _execute_buyback(self, amount: float):
        """Execute AI10X token buyback"""
        try:
            logger.info(f"Executing AI10X buyback of ${amount:.2f}")

            # Record buyback transaction
            buyback_trade = Trade(
                token_address="AI10X_TOKEN_ADDRESS",  # Replace with actual token address
                token_symbol="AI10X",
                price=self._get_token_price(),
                amount=amount,
                trade_type='BUYBACK',
                timestamp=datetime.utcnow()
            )

            db.session.add(buyback_trade)
            db.session.commit()

            self.total_buybacks += amount
            logger.info(f"Buyback executed successfully: ${amount:.2f}")

        except Exception as e:
            logger.error(f"Buyback execution error: {str(e)}")
            db.session.rollback()

    def _add_liquidity(self, amount: float):
        """Add liquidity to AI10X token pool"""
        try:
            logger.info(f"Adding ${amount:.2f} liquidity to AI10X pool")

            # Record liquidity addition
            liquidity_trade = Trade(
                token_address="AI10X_TOKEN_ADDRESS",  # Replace with actual token address
                token_symbol="AI10X",
                price=self._get_token_price(),
                amount=amount,
                trade_type='ADD_LIQUIDITY',
                timestamp=datetime.utcnow()
            )

            db.session.add(liquidity_trade)
            db.session.commit()

            self.total_liquidity_added += amount
            logger.info(f"Liquidity added successfully: ${amount:.2f}")

        except Exception as e:
            logger.error(f"Liquidity addition error: {str(e)}")
            db.session.rollback()

    def _execute_gpu_upgrade(self, amount: float):
        """Execute GPU upgrade with accumulated funds"""
        try:
            logger.info(f"Executing GPU upgrade with ${amount:.2f}")

            # Record GPU upgrade transaction
            upgrade_trade = Trade(
                token_address="AI10X_TOKEN_ADDRESS",  # Replace with actual token address
                token_symbol="AI10X",
                price=self._get_token_price(),
                amount=amount,
                trade_type='GPU_UPGRADE',
                timestamp=datetime.utcnow()
            )

            db.session.add(upgrade_trade)
            db.session.commit()

            self.total_gpu_upgrades += amount
            logger.info(f"GPU upgrade executed successfully: ${amount:.2f}")

        except Exception as e:
            logger.error(f"GPU upgrade error: {str(e)}")
            db.session.rollback()

    def _allocate_trading_capital(self, amount: float):
        """Allocate profits to trading capital"""
        try:
            logger.info(f"Allocating ${amount:.2f} to trading capital")
            # Here we would update the trading capital pool
            # For now, we just log the allocation

        except Exception as e:
            logger.error(f"Trading capital allocation error: {str(e)}")

    def _get_token_price(self) -> float:
        """Get current AI10X token price"""
        # Placeholder - implement actual price fetching
        return 1.0

    def get_allocation_status(self) -> Dict[str, Any]:
        """Get current allocation status"""
        return {
            'accumulated': self.accumulated,
            'total_profits': self.total_profits,
            'total_buybacks': self.total_buybacks,
            'total_liquidity_added': self.total_liquidity_added,
            'total_gpu_upgrades': self.total_gpu_upgrades,
            'thresholds': {
                'buyback': self.min_buyback_amount,
                'liquidity': self.min_liquidity_amount,
                'gpu': self.min_gpu_upgrade
            }
        }

    def get_recent_buybacks(self, limit: int = 25) -> List[Dict[str, Any]]:
        """Get recent buyback operations with success rates"""
        try:
            # Query recent buyback trades
            buyback_trades = Trade.query.filter_by(
                trade_type='BUYBACK'
            ).order_by(Trade.timestamp.desc()).limit(limit).all()

            return [{
                'timestamp': trade.timestamp,
                'amount': trade.amount,
                'success_rate': 1.0 if trade.amount > 0 else 0.0,
                'price': trade.price
            } for trade in buyback_trades] if buyback_trades else []

        except Exception as e:
            logger.error(f"Error getting recent buybacks: {str(e)}")
            return []

    def get_liquidity_metrics(self) -> Dict[str, Any]:
        """Get liquidity operation metrics"""
        try:
            # Query recent liquidity trades
            liquidity_trades = Trade.query.filter_by(
                trade_type='ADD_LIQUIDITY'
            ).order_by(Trade.timestamp.desc()).limit(25).all()

            if not liquidity_trades:
                return {'efficiency': 0.5}

            total_amount = sum(trade.amount for trade in liquidity_trades)
            successful_trades = sum(1 for trade in liquidity_trades if trade.amount > 0)

            return {
                'efficiency': successful_trades / len(liquidity_trades),
                'total_liquidity_added': total_amount,
                'trade_count': len(liquidity_trades)
            }

        except Exception as e:
            logger.error(f"Error getting liquidity metrics: {str(e)}")
            return {'efficiency': 0.5}