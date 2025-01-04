import logging
import time
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

class SolanaMonitor:
    def __init__(self):
        """Initialize Solana monitoring system with HTTP endpoints"""
        self.endpoints = [
            "https://api.mainnet-beta.solana.com",
            "https://solana-api.projectserum.com"
        ]
        self.current_endpoint = 0

        # Performance metrics
        self.metrics = {
            'transaction_success_rate': [],
            'average_confirmation_time': [],
            'slot_distance': [],
            'network_health': [],
            'liquidity_depth': []
        }

        # Monitoring thresholds - set aggressive autonomous thresholds
        self.thresholds = {
            'min_success_rate': 0.85,  # More aggressive threshold
            'max_confirmation_time': 30,  # Faster expected confirmation
            'max_slot_distance': 50,  # Tighter slot monitoring
            'min_health_score': 0.7,  # More aggressive health threshold
            'min_liquidity_depth': 500  # Lower liquidity requirement
        }

        # Error tracking
        self.error_counts = {
            'transaction_failures': 0,
            'network_timeouts': 0,
            'confirmation_delays': 0
        }

        self.last_health_check = datetime.now()
        self.health_check_interval = timedelta(minutes=2)  # More frequent health checks

        logger.info("Solana monitoring system initialized in autonomous mode")

    def _make_request(self, method: str, params: List = None) -> Optional[Dict]:
        """Make HTTP request to Solana RPC endpoint"""
        try:
            headers = {'Content-Type': 'application/json'}
            data = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": method,
                "params": params or []
            }

            response = requests.post(
                self.endpoints[self.current_endpoint],
                json=data,
                headers=headers,
                timeout=10
            )

            if response.status_code == 200:
                return response.json()

            # Switch endpoint on failure
            self._switch_endpoint()
            return None

        except Exception as e:
            logger.error(f"RPC request failed: {str(e)}")
            self._switch_endpoint()
            return None

    def check_network_health(self) -> Dict[str, Any]:
        """Check overall Solana network health using HTTP endpoints"""
        try:
            if (datetime.now() - self.last_health_check) < self.health_check_interval:
                return self._get_latest_health_metrics()

            health_metrics = {
                'network_status': 'healthy',
                'average_confirmation_time': 0,
                'success_rate': 0,
                'current_slot': 0,
                'validator_health': 0
            }

            # Get current slot
            slot_response = self._make_request("getSlot")
            if slot_response and 'result' in slot_response:
                health_metrics['current_slot'] = slot_response['result']

            # Calculate health score
            health_score = self._calculate_health_score(health_metrics)
            health_metrics['health_score'] = health_score
            self.metrics['network_health'].append(health_score)
            self.last_health_check = datetime.now()

            return health_metrics

        except Exception as e:
            logger.error(f"Network health check error: {str(e)}")
            return {
                'network_status': 'degraded',
                'error': str(e)
            }

    def _switch_endpoint(self):
        """Switch to next available Solana RPC endpoint"""
        self.current_endpoint = (self.current_endpoint + 1) % len(self.endpoints)
        logger.info(f"Switched to Solana RPC endpoint: {self.endpoints[self.current_endpoint]}")

    def _calculate_health_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate health score based on available metrics"""
        try:
            # Simplified scoring based on available HTTP metrics
            score_components = []

            if metrics.get('current_slot'):
                score_components.append(1.0)

            if metrics.get('average_confirmation_time', 100) < self.thresholds['max_confirmation_time']:
                score_components.append(1.0)

            return sum(score_components) / max(len(score_components), 1)

        except Exception as e:
            logger.error(f"Error calculating health score: {str(e)}")
            return 0.5

    def _get_latest_health_metrics(self) -> Dict[str, Any]:
        """Get latest cached health metrics"""
        return {
            'network_status': 'healthy' if self.metrics['network_health'] and 
                            self.metrics['network_health'][-1] >= self.thresholds['min_health_score'] else 'degraded',
            'health_score': self.metrics['network_health'][-1] if self.metrics['network_health'] else 0,
            'cached': True,
            'last_check': self.last_health_check.isoformat()
        }

    def monitor_liquidity(self, pool_address: str) -> Dict[str, Any]:
        """Monitor liquidity pool depth using HTTP endpoints"""
        try:
            # Get account info
            response = self._make_request(
                "getAccountInfo",
                [pool_address, {"encoding": "jsonParsed"}]
            )

            if not response or 'result' not in response:
                raise Exception(f"Failed to get pool info for {pool_address}")

            account_data = response['result']
            lamports = account_data.get('value', {}).get('lamports', 0)
            liquidity_depth = float(lamports) / 1e9  # Convert lamports to SOL

            self.metrics['liquidity_depth'].append(liquidity_depth)

            return {
                'address': pool_address,
                'liquidity_depth': liquidity_depth,
                'health_status': 'healthy' if liquidity_depth >= self.thresholds['min_liquidity_depth'] else 'low'
            }

        except Exception as e:
            logger.error(f"Liquidity monitoring error: {str(e)}")
            return {
                'address': pool_address,
                'error': str(e),
                'health_status': 'unknown'
            }

    def monitor_transaction(self, tx_signature: str) -> Dict[str, Any]:
        """Monitor a specific transaction and collect metrics (HTTP version)"""
        try:
            start_time = time.time()
            response = self._make_request("getTransaction", [tx_signature, {"encoding": "jsonParsed"}])
            if not response or 'result' not in response:
                raise Exception(f"Failed to get transaction status for {tx_signature}")

            confirmation_time = time.time() - start_time
            slot_info = self._make_request("getSlot")
            current_slot = slot_info['result'] if slot_info and 'result' in slot_info else 0
            tx_slot = response['result']['slot'] if 'slot' in response['result'] else 0
            slot_distance = current_slot - tx_slot

            self.metrics['transaction_success_rate'].append(1.0)
            self.metrics['average_confirmation_time'].append(confirmation_time)
            self.metrics['slot_distance'].append(slot_distance)
            self._trim_metrics_history()

            return {
                'status': 'confirmed',
                'confirmation_time': confirmation_time,
                'slot_distance': slot_distance,
                'signature': tx_signature
            }

        except Exception as e:
            logger.error(f"Transaction monitoring error: {str(e)}")
            self.error_counts['transaction_failures'] += 1
            self.metrics['transaction_success_rate'].append(0.0)
            return {
                'status': 'failed',
                'error': str(e),
                'signature': tx_signature
            }


    def get_monitoring_metrics(self) -> Dict[str, Any]:
        """Get comprehensive monitoring metrics"""
        try:
            current_metrics = {
                'performance': {
                    'success_rate': sum(self.metrics['transaction_success_rate'][-100:]) / len(self.metrics['transaction_success_rate'][-100:]) if self.metrics['transaction_success_rate'] else 0,
                    'avg_confirmation_time': sum(self.metrics['average_confirmation_time'][-100:]) / len(self.metrics['average_confirmation_time'][-100:]) if self.metrics['average_confirmation_time'] else 0,
                    'avg_slot_distance': sum(self.metrics['slot_distance'][-100:]) / len(self.metrics['slot_distance'][-100:]) if self.metrics['slot_distance'] else 0
                },
                'health': {
                    'network_score': self.metrics['network_health'][-1] if self.metrics['network_health'] else 0,
                    'liquidity_depth': self.metrics['liquidity_depth'][-1] if self.metrics['liquidity_depth'] else 0
                },
                'errors': self.error_counts,
                'thresholds': self.thresholds
            }

            # Add status indicators
            current_metrics['status'] = {
                'performance': 'healthy' if current_metrics['performance']['success_rate'] >= self.thresholds['min_success_rate'] else 'degraded',
                'confirmation': 'healthy' if current_metrics['performance']['avg_confirmation_time'] <= self.thresholds['max_confirmation_time'] else 'slow',
                'network': 'healthy' if current_metrics['health']['network_score'] >= self.thresholds['min_health_score'] else 'degraded',
                'liquidity': 'healthy' if current_metrics['health']['liquidity_depth'] >= self.thresholds['min_liquidity_depth'] else 'low'
            }

            return current_metrics

        except Exception as e:
            logger.error(f"Error getting monitoring metrics: {str(e)}")
            return {
                'error': str(e),
                'status': 'unknown'
            }

    def _trim_metrics_history(self, max_size: int = 1000):
        """Trim metrics history to prevent memory bloat"""
        for metric_type in self.metrics:
            if len(self.metrics[metric_type]) > max_size:
                self.metrics[metric_type] = self.metrics[metric_type][-max_size:]