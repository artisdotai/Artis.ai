"""ETH monitoring module for AI10X platform"""
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from web3 import Web3

logger = logging.getLogger(__name__)

class ETHMonitor:
    """Monitors ETH network conditions and transaction status"""
    def __init__(self):
        # Use Sepolia testnet endpoint
        self.endpoint = f"https://eth-sepolia.g.alchemy.com/v2/{os.environ.get('ALCHEMY_API_KEY')}"
        self.web3 = Web3(Web3.HTTPProvider(self.endpoint))
        self.last_block = None
        self.gas_history = []
        self.gas_window = 10  # Keep last 10 gas price readings

        # Verify connection on initialization
        if not self.web3.is_connected():
            logger.error("Failed to connect to Ethereum network")
        else:
            logger.info("Successfully connected to Ethereum Sepolia network")

    def get_network_status(self) -> Dict[str, Any]:
        """Get current network status and conditions"""
        try:
            current_block = self.web3.eth.block_number
            gas_price = self.web3.eth.gas_price

            # Store gas price history
            self.gas_history.append({
                'price': gas_price,
                'timestamp': datetime.utcnow()
            })
            if len(self.gas_history) > self.gas_window:
                self.gas_history.pop(0)

            # Calculate average and trend
            if len(self.gas_history) > 1:
                avg_gas = sum(h['price'] for h in self.gas_history) / len(self.gas_history)
                gas_trend = (gas_price - self.gas_history[0]['price']) / self.gas_history[0]['price']
            else:
                avg_gas = gas_price
                gas_trend = 0

            network_status = {
                'connected': self.web3.is_connected(),
                'current_block': current_block,
                'blocks_since_last': current_block - self.last_block if self.last_block else 0,
                'network': 'sepolia',
                'gas_price': gas_price,
                'gas_price_gwei': self.web3.from_wei(gas_price, 'gwei'),
                'average_gas_gwei': self.web3.from_wei(avg_gas, 'gwei'),
                'gas_trend': gas_trend,
                'timestamp': datetime.utcnow().isoformat()
            }

            self.last_block = current_block
            return network_status

        except Exception as e:
            logger.error(f"Error getting network status: {str(e)}")
            return {
                'connected': False,
                'error': str(e),
                'network': 'sepolia',
                'timestamp': datetime.utcnow().isoformat()
            }

    def check_transaction_status(self, tx_hash: str) -> Dict[str, Any]:
        """Get detailed transaction status and confirmations"""
        try:
            tx_receipt = self.web3.eth.get_transaction_receipt(tx_hash)
            tx_details = self.web3.eth.get_transaction(tx_hash)

            if not tx_receipt:
                return {
                    'status': 'pending',
                    'confirmations': 0,
                    'timestamp': datetime.utcnow().isoformat()
                }

            current_block = self.web3.eth.block_number
            confirmations = current_block - tx_receipt.blockNumber

            return {
                'status': 'success' if tx_receipt.status else 'failed',
                'confirmations': confirmations,
                'block_number': tx_receipt.blockNumber,
                'gas_used': tx_receipt.gasUsed,
                'effective_gas_price': tx_receipt.effectiveGasPrice,
                'value': self.web3.from_wei(tx_details.value, 'ether'),
                'timestamp': datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Error checking transaction status: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }

    def get_gas_price_stats(self) -> Dict[str, Any]:
        """Get gas price statistics and recommendations"""
        try:
            current_gas = self.web3.eth.gas_price

            # Calculate percentiles from history
            if self.gas_history:
                prices = [h['price'] for h in self.gas_history]
                prices.sort()

                # Simple percentile calculation
                def get_percentile(p):
                    k = (len(prices) - 1) * p
                    f = int(k)
                    c = int(k + 1 if k + 1 < len(prices) else k)
                    d = k - f
                    return prices[f] * (1 - d) + prices[c] * d

                return {
                    'current': self.web3.from_wei(current_gas, 'gwei'),
                    'slow': self.web3.from_wei(get_percentile(0.2), 'gwei'),
                    'standard': self.web3.from_wei(get_percentile(0.5), 'gwei'),
                    'fast': self.web3.from_wei(get_percentile(0.8), 'gwei'),
                    'network': 'sepolia',
                    'timestamp': datetime.utcnow().isoformat()
                }
            else:
                return {
                    'current': self.web3.from_wei(current_gas, 'gwei'),
                    'slow': self.web3.from_wei(current_gas * 0.8, 'gwei'),
                    'standard': self.web3.from_wei(current_gas, 'gwei'),
                    'fast': self.web3.from_wei(current_gas * 1.2, 'gwei'),
                    'network': 'sepolia',
                    'timestamp': datetime.utcnow().isoformat()
                }

        except Exception as e:
            logger.error(f"Error getting gas price stats: {str(e)}")
            return {
                'error': str(e),
                'network': 'sepolia',
                'timestamp': datetime.utcnow().isoformat()
            }