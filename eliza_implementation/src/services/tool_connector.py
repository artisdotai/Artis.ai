"""
Tool connector service for Eliza Framework implementation with enhanced error handling
"""
import logging
import os
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
import json
from functools import lru_cache

logger = logging.getLogger(__name__)

class RateLimiter:
    """Rate limiter implementation for API calls"""
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = []

    def check_rate_limit(self) -> bool:
        """Check if current request is within rate limits"""
        now = datetime.utcnow()
        self.requests = [ts for ts in self.requests 
                        if (now - ts).total_seconds() < self.window_seconds]

        if len(self.requests) >= self.max_requests:
            return False

        self.requests.append(now)
        return True

class CircuitBreaker:
    """Circuit breaker implementation for tool execution"""
    def __init__(self, failure_threshold: int = 5, reset_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failures = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
        self.telemetry = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'last_state_change': datetime.utcnow(),
            'uptime_percentage': 100.0
        }

    def can_execute(self) -> bool:
        """Check if circuit breaker allows execution"""
        self.telemetry['total_requests'] += 1

        if self.state == "closed":
            return True

        if self.state == "open":
            if self.last_failure_time and (datetime.utcnow() - self.last_failure_time).total_seconds() > self.reset_timeout:
                self._change_state("half-open")
                return True
            return False

        return True  # half-open state allows one test request

    def record_success(self) -> None:
        """Record successful execution"""
        self.telemetry['successful_requests'] += 1

        if self.state == "half-open":
            self._change_state("closed")
        self.failures = 0
        self.last_failure_time = None
        self._update_uptime()

    def record_failure(self) -> None:
        """Record failed execution"""
        self.failures += 1
        self.telemetry['failed_requests'] += 1
        self.last_failure_time = datetime.utcnow()

        if self.failures >= self.failure_threshold:
            self._change_state("open")
        self._update_uptime()

    def _change_state(self, new_state: str) -> None:
        """Update circuit breaker state with telemetry"""
        self.state = new_state
        self.telemetry['last_state_change'] = datetime.utcnow()
        logger.info(f"Circuit breaker state changed to: {new_state}")

    def _update_uptime(self) -> None:
        """Update service uptime percentage"""
        total = self.telemetry['total_requests']
        if total > 0:
            self.telemetry['uptime_percentage'] = (
                self.telemetry['successful_requests'] / total
            ) * 100

    def get_telemetry(self) -> Dict[str, Any]:
        """Get current telemetry data"""
        return {
            **self.telemetry,
            'current_state': self.state,
            'current_failures': self.failures,
            'last_failure': self.last_failure_time.isoformat() if self.last_failure_time else None
        }

class ElizaTool:
    """Base class for ELIZA tools with enhanced error handling"""
    def __init__(self, name: str, description: str, parameters: List[str], handler: Callable):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.handler = handler
        self.initialized = False
        self.last_check = None
        self.error_message = None
        self.validation_details = {}
        self._cache = {}
        self._cache_timeout = 300  # 5 minutes
        self.circuit_breaker = CircuitBreaker()
        self.rate_limiter = RateLimiter()

    def _get_cache_key(self, parameters: Dict[str, Any]) -> str:
        """Generate cache key from parameters"""
        return f"{self.name}:{json.dumps(parameters, sort_keys=True)}"

    def _is_cache_valid(self, cached_result: Dict[str, Any]) -> bool:
        """Check if cached result is still valid"""
        if not cached_result or 'timestamp' not in cached_result:
            return False
        return (datetime.utcnow().timestamp() - cached_result['timestamp']) < self._cache_timeout

    def _validate_parameters(self, parameters: Dict[str, Any]) -> Optional[str]:
        """Validate input parameters"""
        if not parameters:
            return "No parameters provided"

        missing_params = set(self.parameters) - set(parameters.keys())
        if missing_params:
            return f"Missing required parameters: {', '.join(missing_params)}"

        return None

    def execute_with_cache(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute handler with caching and circuit breaker support"""
        try:
            # Validate parameters
            validation_error = self._validate_parameters(parameters)
            if validation_error:
                return {
                    'status': 'error',
                    'message': validation_error,
                    'timestamp': datetime.utcnow().isoformat()
                }

            # Check rate limits
            if not self.rate_limiter.check_rate_limit():
                return {
                    'status': 'error',
                    'message': 'Rate limit exceeded',
                    'timestamp': datetime.utcnow().isoformat()
                }

            # Check circuit breaker
            if not self.circuit_breaker.can_execute():
                return {
                    'status': 'error',
                    'message': f'Service {self.name} is currently unavailable',
                    'timestamp': datetime.utcnow().isoformat(),
                    'circuit_breaker': self.circuit_breaker.get_telemetry()
                }

            cache_key = self._get_cache_key(parameters)

            # Check cache
            if cache_key in self._cache and self._is_cache_valid(self._cache[cache_key]):
                cached_result = self._cache[cache_key]
                logger.info(f"Cache hit for {self.name}")
                return {
                    **cached_result['result'],
                    'cache_status': 'hit',
                    'cached_at': datetime.fromtimestamp(cached_result['timestamp']).isoformat()
                }

            # Execute handler
            result = self.handler(parameters)
            if result and isinstance(result, dict):
                # Add execution timestamp
                if 'timestamp' not in result:
                    result['timestamp'] = datetime.utcnow().isoformat()

                # Cache successful result
                if result.get('status') != 'error':
                    self._cache[cache_key] = {
                        'result': result,
                        'timestamp': datetime.utcnow().timestamp()
                    }
                    self.circuit_breaker.record_success()

                result['cache_status'] = 'miss'
                return result

            return {
                'status': 'error',
                'message': 'Invalid handler response',
                'timestamp': datetime.utcnow().isoformat(),
                'cache_status': 'miss'
            }

        except Exception as e:
            logger.error(f"Handler execution error: {str(e)}")
            self.circuit_breaker.record_failure()
            return {
                'status': 'error',
                'message': str(e),
                'timestamp': datetime.utcnow().isoformat(),
                'cache_status': 'miss',
                'circuit_breaker': self.circuit_breaker.get_telemetry()
            }

    def validate(self) -> bool:
        """Validate tool configuration and dependencies"""
        try:
            # Basic validation
            if not all([self.name, self.description, self.parameters, self.handler]):
                raise ValueError("Missing required tool configuration")

            if not callable(self.handler):
                raise ValueError("Tool handler must be callable")

            self.initialized = True
            self.last_check = datetime.utcnow()
            self.validation_details = {
                'last_check': self.last_check.isoformat(),
                'status': 'operational',
                'message': 'All validations passed',
                'circuit_breaker': self.circuit_breaker.get_telemetry()
            }
            return True

        except Exception as e:
            self.error_message = str(e)
            self.validation_details = {
                'last_check': datetime.utcnow().isoformat(),
                'status': 'error',
                'message': str(e),
                'circuit_breaker': self.circuit_breaker.get_telemetry()
            }
            logger.error(f"Tool validation error for {self.name}: {str(e)}")
            return False

class ToolConnector:
    """Tool connector service implementation"""

    def __init__(self):
        """Initialize tool connector"""
        try:
            # Tool registry
            self.tools: Dict[str, ElizaTool] = {}
            self._register_core_tools()
            logger.info("Tool connector initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing tool connector: {str(e)}")
            raise

    def _register_core_tools(self):
        """Register all core ELIZA tools"""
        try:
            # Market Analysis Tool
            self.register_tool(
                "market_analysis",
                "Analyzes market conditions and trends",
                ["chain", "metrics"],
                self._execute_market_analysis
            )

            # Sentiment Analysis Tool
            self.register_tool(
                "sentiment_analysis", 
                "Analyzes market sentiment from social data",
                ["source", "content"],
                self._execute_sentiment_analysis
            )

            # Risk Management Tool
            self.register_tool(
                "risk_management",
                "Manages trading risk parameters",
                ["risk_level", "parameters"],
                self._execute_risk_management
            )

            # LLM Analysis Tool
            self.register_tool(
                "llm_analysis",
                "Performs LLM-based analysis",
                ["content", "analysis_type"],
                self._execute_llm_analysis
            )

            # Health Monitoring Tool
            self.register_tool(
                "health_monitoring",
                "Monitors system health and status",
                ["component", "metrics"],
                self._execute_health_monitoring
            )

            # Trading Execution Tool
            self.register_tool(
                "trading_execution",
                "Executes trading operations",
                ["chain", "parameters"],
                self._execute_trading
            )

            # Twitter Analysis Tool
            self.register_tool(
                "twitter_analysis",
                "Analyzes Twitter data for crypto signals",
                ["query", "time_range", "metrics"],
                self._execute_twitter_analysis
            )

            # GMGN Integration Tool
            self.register_tool(
                "gmgn_integration",
                "Integrates with GMGN API for historical data",
                ["token_address", "time_range", "metrics"],
                self._execute_gmgn_analysis
            )

            # PumpFun Integration Tool
            self.register_tool(
                "pumpfun_integration",
                "Monitors new token launches via PumpFun",
                ["chain", "filters", "metrics"],
                self._execute_pumpfun_analysis
            )

            # SolanaSniffer Integration Tool
            self.register_tool(
                "solana_sniffer",
                "Verifies token safety using SolanaSniffer",
                ["token_address", "safety_metrics"],
                self._execute_solana_sniffer
            )

            # Pattern Recognition Tool
            self.register_tool(
                "pattern_recognition",
                "Identifies trading patterns and signals",
                ["token_data", "pattern_types"],
                self._execute_pattern_recognition
            )

            # Portfolio Optimization Tool
            self.register_tool(
                "portfolio_optimization",
                "Optimizes trading portfolio allocation",
                ["positions", "risk_parameters"],
                self._execute_portfolio_optimization
            )

            # Historical Analysis Tool
            self.register_tool(
                "historical_analysis",
                "Analyzes historical trading performance",
                ["token_address", "time_range", "metrics"],
                self._execute_historical_analysis
            )

            # Event Impact Tool
            self.register_tool(
                "event_impact",
                "Analyzes market events impact",
                ["event_data", "impact_metrics"],
                self._execute_event_impact
            )

        except Exception as e:
            logger.error(f"Error registering core tools: {str(e)}")
            raise

    def register_tool(self, tool_id: str, description: str, parameters: List[str], 
                     handler: Callable) -> bool:
        """Register a new tool"""
        try:
            if tool_id in self.tools:
                logger.warning(f"Tool {tool_id} already registered")
                return False

            tool = ElizaTool(tool_id, description, parameters, handler)
            if tool.validate():
                self.tools[tool_id] = tool
                logger.info(f"Successfully registered tool: {tool_id}")
                return True

            logger.error(f"Failed to validate tool: {tool_id}")
            return False

        except Exception as e:
            logger.error(f"Error registering tool {tool_id}: {str(e)}")
            return False

    def get_available_tools(self) -> Dict[str, Any]:
        """Get list of available tools and their configurations"""
        try:
            return {
                'status': 'success',
                'tools': {
                    tool_id: {
                        'name': tool.name,
                        'description': tool.description,
                        'parameters': tool.parameters,
                        'status': 'operational' if tool.initialized else 'error',
                        'last_check': tool.last_check.isoformat() if tool.last_check else None,
                        'error': tool.error_message,
                        'validation_details': tool.validation_details
                    } for tool_id, tool in self.tools.items()
                },
                'timestamp': datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting available tools: {str(e)}")
            return {
                'status': 'error',
                'message': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }

    def execute_tool(self, tool_id: str, parameters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Execute specific tool with given parameters"""
        try:
            if tool_id not in self.tools:
                raise ValueError(f"Tool not found: {tool_id}")

            tool = self.tools[tool_id]

            # Validate tool status
            if not tool.initialized:
                if not tool.validate():
                    raise ValueError(f"Tool {tool_id} failed validation: {tool.error_message}")

            # Execute tool handler with caching
            return tool.execute_with_cache(parameters)

        except Exception as e:
            logger.error(f"Error executing tool {tool_id}: {str(e)}")
            return {
                'status': 'error',
                'message': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }

    def _execute_market_analysis(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute market analysis tool"""
        from .llm_analysis import LLMAnalyzer

        try:
            analyzer = LLMAnalyzer()
            result = analyzer.analyze_market_trends(
                parameters['chain'],
                parameters['metrics']
            )

            if result:
                result['tool_id'] = 'market_analysis'
                result['parameters'] = parameters
                return result

            return {
                'status': 'error',
                'message': 'Analysis failed',
                'tool_id': 'market_analysis',
                'timestamp': datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Market analysis error: {str(e)}")
            return {
                'status': 'error',
                'message': str(e),
                'tool_id': 'market_analysis',
                'timestamp': datetime.utcnow().isoformat()
            }

    def _execute_sentiment_analysis(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute sentiment analysis tool"""
        from .sentiment import SentimentAnalyzer

        try:
            analyzer = SentimentAnalyzer()
            result = analyzer.analyze_content(
                parameters['content'],
                parameters['source']
            )

            if result:
                result['tool_id'] = 'sentiment_analysis'
                result['parameters'] = parameters
                return result

            return {
                'status': 'error',
                'message': 'Analysis failed',
                'tool_id': 'sentiment_analysis',
                'timestamp': datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Sentiment analysis error: {str(e)}")
            return {
                'status': 'error',
                'message': str(e),
                'tool_id': 'sentiment_analysis',
                'timestamp': datetime.utcnow().isoformat()
            }

    def _execute_risk_management(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute risk management tool"""
        from .risk_management import RiskManager

        try:
            manager = RiskManager()
            risk_params = manager.get_current_parameters()

            return {
                'parameters': risk_params,
                'status': 'success',
                'tool_id': 'risk_management',
                'request_parameters': parameters,
                'timestamp': datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Risk management error: {str(e)}")
            return {
                'status': 'error',
                'message': str(e),
                'tool_id': 'risk_management',
                'timestamp': datetime.utcnow().isoformat()
            }

    def _execute_llm_analysis(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute LLM analysis tool"""
        from .llm_analysis import LLMAnalyzer

        try:
            analyzer = LLMAnalyzer()
            result = analyzer.analyze_market_trends(
                parameters.get('chain', 'ETH'),
                parameters.get('metrics', {})
            )

            if result:
                result['tool_id'] = 'llm_analysis'
                result['parameters'] = parameters
                return result

            return {
                'status': 'error',
                'message': 'Analysis failed',
                'tool_id': 'llm_analysis',
                'timestamp': datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"LLM analysis error: {str(e)}")
            return {
                'status': 'error',
                'message': str(e),
                'tool_id': 'llm_analysis',
                'timestamp': datetime.utcnow().isoformat()
            }

    def _execute_health_monitoring(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute health monitoring tool"""
        from .monitoring import HealthMonitor

        try:
            monitor = HealthMonitor()
            result = monitor.get_system_status()

            if result:
                result['tool_id'] = 'health_monitoring'
                result['parameters'] = parameters
                return result

            return {
                'status': 'error',
                'message': 'Health check failed',
                'tool_id': 'health_monitoring',
                'timestamp': datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Health monitoring error: {str(e)}")
            return {
                'status': 'error',
                'message': str(e),
                'tool_id': 'health_monitoring',
                'timestamp': datetime.utcnow().isoformat()
            }

    def _execute_trading(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute trading tool"""
        from .trading import TradingService

        try:
            trading = TradingService()
            result = trading.execute_trade(
                parameters['chain'],
                parameters
            )

            if result:
                result['tool_id'] = 'trading_execution'
                result['parameters'] = parameters
                return result

            return {
                'status': 'error',
                'message': 'Trade execution failed',
                'tool_id': 'trading_execution',
                'timestamp': datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Trading error: {str(e)}")
            return {
                'status': 'error',
                'message': str(e),
                'tool_id': 'trading_execution',
                'timestamp': datetime.utcnow().isoformat()
            }

    def _execute_twitter_analysis(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Twitter analysis tool"""
        from .twitter_analyzer import TwitterAnalyzer

        try:
            analyzer = TwitterAnalyzer()
            result = analyzer.analyze_signals(
                parameters['query'],
                parameters['time_range'],
                parameters['metrics']
            )

            if result:
                result['tool_id'] = 'twitter_analysis'
                result['parameters'] = parameters
                return result

            return {
                'status': 'error',
                'message': 'Twitter analysis failed',
                'tool_id': 'twitter_analysis',
                'timestamp': datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Twitter analysis error: {str(e)}")
            return {
                'status': 'error',
                'message': str(e),
                'tool_id': 'twitter_analysis',
                'timestamp': datetime.utcnow().isoformat()
            }

    def _execute_gmgn_analysis(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute GMGN integration tool"""
        from .gmgn_analyzer import GMGNAnalyzer

        try:
            analyzer = GMGNAnalyzer()
            result = analyzer.analyze_token(
                parameters['token_address'],
                parameters['time_range'],
                parameters['metrics']
            )

            if result:
                result['tool_id'] = 'gmgn_integration'
                result['parameters'] = parameters
                return result

            return {
                'status': 'error',
                'message': 'GMGN analysis failed',
                'tool_id': 'gmgn_integration',
                'timestamp': datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"GMGN analysis error: {str(e)}")
            return {
                'status': 'error',
                'message': str(e),
                'tool_id': 'gmgn_integration',
                'timestamp': datetime.utcnow().isoformat()
            }

    def _execute_pumpfun_analysis(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute PumpFun integration tool"""
        from .pumpfun_analyzer import PumpFunAnalyzer

        try:
            analyzer = PumpFunAnalyzer()
            result = analyzer.monitor_launches(
                parameters['chain'],
                parameters['filters'],
                parameters['metrics']
            )

            if result:
                result['tool_id'] = 'pumpfun_integration'
                result['parameters'] = parameters
                return result

            return {
                'status': 'error',
                'message': 'PumpFun analysis failed',
                'tool_id': 'pumpfun_integration',
                'timestamp': datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"PumpFun analysis error: {str(e)}")
            return {
                'status': 'error',
                'message': str(e),
                'tool_id': 'pumpfun_integration',
                'timestamp': datetime.utcnow().isoformat()
            }

    def _execute_solana_sniffer(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute SolanaSniffer integration tool"""
        from .solana_sniffer import SolanaSniffer

        try:
            analyzer = SolanaSniffer()
            result = analyzer.verify_token(
                parameters['token_address'],
                parameters['safety_metrics']
            )

            if result:
                result['tool_id'] = 'solana_sniffer'
                result['parameters'] = parameters
                return result

            return {
                'status': 'error',
                'message': 'SolanaSniffer verification failed',
                'tool_id': 'solana_sniffer',
                'timestamp': datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"SolanaSniffer error: {str(e)}")
            return {
                'status': 'error',
                'message': str(e),
                'tool_id': 'solana_sniffer',
                'timestamp': datetime.utcnow().isoformat()
            }

    def _execute_pattern_recognition(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute pattern recognition tool"""
        from .pattern_recognition import PatternRecognizer

        try:
            analyzer = PatternRecognizer()
            result = analyzer.identify_patterns(
                parameters['token_data'],
                parameters['pattern_types']
            )

            if result:
                result['tool_id'] = 'pattern_recognition'
                result['parameters'] = parameters
                return result

            return {
                'status': 'error',
                'message': 'Pattern recognition failed',
                'tool_id': 'pattern_recognition',
                'timestamp': datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Pattern recognition error: {str(e)}")
            return {
                'status': 'error',
                'message': str(e),
                'tool_id': 'pattern_recognition',
                'timestamp': datetime.utcnow().isoformat()
            }

    def _execute_portfolio_optimization(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute portfolio optimization tool"""
        from .portfolio_optimizer import PortfolioOptimizer

        try:
            optimizer = PortfolioOptimizer()
            result = optimizer.optimize_allocation(
                parameters['positions'],
                parameters['risk_parameters']
            )

            if result:
                result['tool_id'] = 'portfolio_optimization'
                result['parameters'] = parameters
                return result

            return {
                'status': 'error',
                'message': 'Portfolio optimization failed',
                'tool_id': 'portfolio_optimization',
                'timestamp': datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Portfolio optimization error: {str(e)}")
            return {
                'status': 'error',
                'message': str(e),
                'tool_id': 'portfolio_optimization',
                'timestamp': datetime.utcnow().isoformat()
            }

    def _execute_historical_analysis(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute historical analysis tool"""
        from .historical_analyzer import HistoricalAnalyzer

        try:
            analyzer = HistoricalAnalyzer()
            result = analyzer.analyze_performance(
                parameters['token_address'],
                parameters['time_range'],
                parameters['metrics']
            )

            if result:
                result['tool_id'] = 'historical_analysis'
                result['parameters'] = parameters
                return result

            return {
                'status': 'error',
                'message': 'Historical analysis failed',
                'tool_id': 'historical_analysis',
                'timestamp': datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Historical analysis error: {str(e)}")
            return {
                'status': 'error',
                'message': str(e),
                'tool_id': 'historical_analysis',
                'timestamp': datetime.utcnow().isoformat()
            }

    def _execute_event_impact(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute event impact analysis tool"""
        from .event_analyzer import EventAnalyzer

        try:
            analyzer = EventAnalyzer()
            result = analyzer.analyze_impact(
                parameters['event_data'],
                parameters['impact_metrics']
            )

            if result:
                result['tool_id'] = 'event_impact'
                result['parameters'] = parameters
                return result

            return {
                'status': 'error',
                'message': 'Event impact analysis failed',
                'tool_id': 'event_impact',
                'timestamp': datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Event impact analysis error: {str(e)}")
            return {
                'status': 'error',
                'message': str(e),
                'tool_id': 'event_impact',
                'timestamp': datetime.utcnow().isoformat()
            }