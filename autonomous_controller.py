import logging
import time
import psutil
from typing import Dict, Any, List, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

class AutonomousController:
    def __init__(self):
        """Initialize autonomous controller with enhanced monitoring"""
        self.auto_mode = True
        self.last_action_time = datetime.now()
        self.action_cooldown = 0.05
        self.confidence_threshold = 0.25
        self.base_confidence_threshold = 0.25

        # Enhanced monitoring
        self.system_metrics = {
            'cpu_usage': [],
            'memory_usage': [],
            'disk_usage': [],
            'network_latency': [],
            'api_response_times': [],
            'error_rates': {}
        }

        # Performance optimization
        self.performance_thresholds = {
            'cpu_high': 80.0,
            'memory_high': 85.0,
            'latency_high': 1000,  # ms
            'error_rate_high': 0.1
        }

        # Auto-adjustment parameters
        self.learning_rate = 0.1
        self.optimization_window = 100  # Number of metrics to consider
        self.min_optimization_interval = 300  # 5 minutes
        self.last_optimization = datetime.now()

        # Track optimization history
        self.optimization_history = []

        # Performance tracking (remaining from original)
        self.success_rate = []
        self.action_history = []
        self.error_counts = {}

        # Auto-adjustment parameters (remaining from original)
        self.min_success_rate = 0.15  # Very low bound for maximum aggression
        self.adjustment_factor = 0.2  # Very large adjustments


        logger.info("Autonomous controller initialized with enhanced monitoring")

    def should_verify(self, action_type: str, confidence: float) -> bool:
        """Always return False - operate fully autonomously"""
        return False

    def can_execute(self, action_type: str) -> bool:
        """Check if enough time has passed since last action"""
        if (datetime.now() - self.last_action_time).total_seconds() < self.action_cooldown:
            return False
        return True

    def record_action(self, action_type: str, success: bool, details: Dict[str, Any] = None):
        """Record action outcome for learning"""
        self.action_history.append({
            'type': action_type,
            'success': success,
            'timestamp': datetime.now(),
            'details': details or {}
        })

        if success:
            self.success_rate.append(1)
        else:
            self.success_rate.append(0)
            self.error_counts[action_type] = self.error_counts.get(action_type, 0) + 1

        # Adjust parameters automatically based on performance
        self._adjust_parameters()

    def monitor_system_health(self) -> Dict[str, Any]:
        """Monitor system health metrics"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_percent = psutil.virtual_memory().percent
            disk_usage = psutil.disk_usage('/').percent

            self.system_metrics['cpu_usage'].append(cpu_percent)
            self.system_metrics['memory_usage'].append(memory_percent)
            self.system_metrics['disk_usage'].append(disk_usage)

            # Keep only recent metrics
            for key in self.system_metrics:
                if len(self.system_metrics[key]) > self.optimization_window:
                    self.system_metrics[key] = self.system_metrics[key][-self.optimization_window:]

            return {
                'cpu_usage': cpu_percent,
                'memory_usage': memory_percent,
                'disk_usage': disk_usage,
                'status': self._evaluate_system_status()
            }
        except Exception as e:
            logger.error(f"Error monitoring system health: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }

    def _evaluate_system_status(self) -> str:
        """Evaluate overall system status based on metrics"""
        try:
            recent_cpu = self.system_metrics['cpu_usage'][-10:]
            recent_memory = self.system_metrics['memory_usage'][-10:]

            avg_cpu = sum(recent_cpu) / len(recent_cpu)
            avg_memory = sum(recent_memory) / len(recent_memory)

            if avg_cpu > self.performance_thresholds['cpu_high'] or \
               avg_memory > self.performance_thresholds['memory_high']:
                return 'critical'
            elif avg_cpu > self.performance_thresholds['cpu_high'] * 0.8 or \
                 avg_memory > self.performance_thresholds['memory_high'] * 0.8:
                return 'warning'
            return 'healthy'
        except Exception as e:
            logger.error(f"Error evaluating system status: {str(e)}")
            return 'unknown'

    def should_optimize(self) -> Tuple[bool, List[str]]:
        """Determine if optimization is needed"""
        try:
            time_since_last = (datetime.now() - self.last_optimization).total_seconds()
            if time_since_last < self.min_optimization_interval:
                return False, []

            reasons = []
            status = self._evaluate_system_status()

            # Check system metrics
            if status in ['critical', 'warning']:
                reasons.append(f"System status: {status}")

            # Check error rates
            for service, errors in self.system_metrics['error_rates'].items():
                error_rate = sum(errors) / len(errors) if errors else 0
                if error_rate > self.performance_thresholds['error_rate_high']:
                    reasons.append(f"High error rate for {service}: {error_rate:.2%}")

            # Check API response times
            if self.system_metrics['api_response_times']:
                avg_latency = sum(self.system_metrics['api_response_times']) / \
                            len(self.system_metrics['api_response_times'])
                if avg_latency > self.performance_thresholds['latency_high']:
                    reasons.append(f"High API latency: {avg_latency:.0f}ms")

            return len(reasons) > 0, reasons

        except Exception as e:
            logger.error(f"Error checking optimization need: {str(e)}")
            return False, [f"Error: {str(e)}"]

    def optimize_system(self) -> Dict[str, Any]:
        """Perform system optimization"""
        try:
            should_optimize, reasons = self.should_optimize()
            if not should_optimize:
                return {'status': 'skipped', 'message': 'Optimization not needed'}

            optimization_results = []

            # Memory optimization
            if psutil.virtual_memory().percent > self.performance_thresholds['memory_high']:
                logger.info("Performing memory optimization")
                optimization_results.append(self._optimize_memory())

            # Performance optimization
            status = self._evaluate_system_status()
            if status in ['critical', 'warning']:
                logger.info(f"Performing performance optimization due to {status} status")
                optimization_results.append(self._optimize_performance())

            # API optimization
            if self.system_metrics['api_response_times']:
                avg_latency = sum(self.system_metrics['api_response_times']) / \
                            len(self.system_metrics['api_response_times'])
                if avg_latency > self.performance_thresholds['latency_high']:
                    logger.info("Performing API optimization")
                    optimization_results.append(self._optimize_api_performance())

            # Record optimization
            self.last_optimization = datetime.now()
            self.optimization_history.append({
                'timestamp': self.last_optimization,
                'reasons': reasons,
                'results': optimization_results
            })

            return {
                'status': 'success',
                'timestamp': self.last_optimization.isoformat(),
                'reasons': reasons,
                'optimizations': optimization_results
            }

        except Exception as e:
            logger.error(f"Error during system optimization: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }

    def _optimize_memory(self) -> Dict[str, Any]:
        """Optimize memory usage"""
        try:
            import gc
            gc.collect()
            return {
                'type': 'memory',
                'status': 'success',
                'message': 'Memory optimization completed'
            }
        except Exception as e:
            logger.error(f"Memory optimization failed: {str(e)}")
            return {
                'type': 'memory',
                'status': 'error',
                'error': str(e)
            }

    def _optimize_performance(self) -> Dict[str, Any]:
        """Optimize system performance"""
        try:
            # Adjust monitoring intervals based on system load
            cpu_load = psutil.cpu_percent()
            if cpu_load > self.performance_thresholds['cpu_high']:
                self.action_cooldown = max(0.1, self.action_cooldown * 1.5)
            else:
                self.action_cooldown = max(0.05, self.action_cooldown * 0.8)

            return {
                'type': 'performance',
                'status': 'success',
                'adjustments': {
                    'action_cooldown': self.action_cooldown
                }
            }
        except Exception as e:
            logger.error(f"Performance optimization failed: {str(e)}")
            return {
                'type': 'performance',
                'status': 'error',
                'error': str(e)
            }

    def _optimize_api_performance(self) -> Dict[str, Any]:
        """Optimize API performance"""
        try:
            # Adjust rate limiting based on response times
            avg_latency = sum(self.system_metrics['api_response_times']) / \
                         len(self.system_metrics['api_response_times'])

            if avg_latency > self.performance_thresholds['latency_high']:
                self.confidence_threshold = min(0.4, self.confidence_threshold * 1.2)
            else:
                self.confidence_threshold = max(0.25, self.confidence_threshold * 0.8)

            return {
                'type': 'api',
                'status': 'success',
                'adjustments': {
                    'confidence_threshold': self.confidence_threshold
                }
            }
        except Exception as e:
            logger.error(f"API optimization failed: {str(e)}")
            return {
                'type': 'api',
                'status': 'error',
                'error': str(e)
            }

    def get_success_rate(self, action_type: str = None) -> float:
        """Get success rate for specific action type or overall"""
        if not self.action_history:
            return 0.7  # Extremely optimistic default

        if action_type:
            relevant_actions = [a for a in self.action_history if a['type'] == action_type]
            if not relevant_actions:
                return 0.7  # Extremely optimistic default
            return sum(1 for a in relevant_actions if a['success']) / len(relevant_actions)

        return sum(self.success_rate) / len(self.success_rate)

    def _adjust_parameters(self):
        """Adjust controller parameters based on performance"""
        recent_success_rate = sum(self.success_rate[-25:]) / len(self.success_rate[-25:]) if self.success_rate else 0.7

        # Adjust base confidence threshold
        if recent_success_rate > 0.4:  # More aggressive threshold
            self.base_confidence_threshold = max(0.15, self.base_confidence_threshold - self.adjustment_factor)
        elif recent_success_rate < self.min_success_rate:  # Still maintain minimal safety
            self.base_confidence_threshold = min(0.4, self.base_confidence_threshold + self.adjustment_factor)

        # Update confidence threshold
        self.confidence_threshold = self.base_confidence_threshold * (1 + (recent_success_rate - 0.5) * self.learning_rate)

        # Adjust action cooldown
        if recent_success_rate > 0.3:  # Ultra-aggressive cooldown adjustment
            self.action_cooldown = max(0.02, self.action_cooldown - 0.1)  # Even faster actions
        else:
            self.action_cooldown = min(0.2, self.action_cooldown + 0.05)

        # Update learning rate based on performance stability
        variance = sum((x - recent_success_rate) ** 2 for x in self.success_rate[-25:]) / 25 if self.success_rate else 0
        if variance < 0.15:  # Less strict stability requirement
            self.learning_rate = min(0.2, self.learning_rate + 0.03)  # Faster learning
        else:
            self.learning_rate = max(0.05, self.learning_rate - 0.02)  # Higher minimum learning rate


    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            health = self.monitor_system_health()
            optimization_needed, reasons = self.should_optimize()

            recent_optimizations = sorted(
                self.optimization_history,
                key=lambda x: x['timestamp'],
                reverse=True
            )[:5]

            return {
                'health': health,
                'optimization_needed': optimization_needed,
                'optimization_reasons': reasons,
                'metrics': {
                    'action_cooldown': self.action_cooldown,
                    'confidence_threshold': self.confidence_threshold,
                    'learning_rate': self.learning_rate,
                    'success_rate': self.get_success_rate(),
                    'error_counts': self.error_counts
                },
                'recent_optimizations': recent_optimizations,
                'system_metrics': {
                    'cpu_usage': self.system_metrics['cpu_usage'][-10:],
                    'memory_usage': self.system_metrics['memory_usage'][-10:],
                    'disk_usage': self.system_metrics['disk_usage'][-10:],
                    'api_latency': self.system_metrics['api_response_times'][-10:]
                }
            }
        except Exception as e:
            logger.error(f"Error getting system status: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }

    def record_api_metrics(self, service: str, success: bool, latency: float = None):
        """Record API performance metrics"""
        try:
            if service not in self.system_metrics['error_rates']:
                self.system_metrics['error_rates'][service] = []

            self.system_metrics['error_rates'][service].append(0 if success else 1)
            if len(self.system_metrics['error_rates'][service]) > self.optimization_window:
                self.system_metrics['error_rates'][service] = \
                    self.system_metrics['error_rates'][service][-self.optimization_window:]

            if latency is not None:
                self.system_metrics['api_response_times'].append(latency)
                if len(self.system_metrics['api_response_times']) > self.optimization_window:
                    self.system_metrics['api_response_times'] = \
                        self.system_metrics['api_response_times'][-self.optimization_window:]

        except Exception as e:
            logger.error(f"Error recording API metrics: {str(e)}")