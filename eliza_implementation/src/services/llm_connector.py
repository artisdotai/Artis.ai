"""
LLM Connector service for Eliza Framework implementation
"""
import os
import logging
import json
from typing import Dict, Any, Optional
from datetime import datetime
import openai
import cohere
from huggingface_hub import InferenceClient

logger = logging.getLogger(__name__)

class LLMConnector:
    """LLM service connector implementation"""

    def __init__(self):
        """Initialize LLM connectors"""
        try:
            # Initialize OpenAI client
            self.openai_client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

            # Initialize Cohere client
            self.cohere_client = cohere.Client(os.environ.get("COHERE_API_KEY"))

            # Initialize HuggingFace client
            self.huggingface_client = InferenceClient(token=os.environ.get("HUGGINGFACE_API_KEY"))

            # The newest OpenAI model is "gpt-4o" which was released May 13, 2024
            # Do not change this unless explicitly requested by the user
            self.models = {
                'openai': 'gpt-4o',
                'cohere': 'command',
                'huggingface': 'mistralai/Mistral-7B-Instruct-v0.2'  # Changed to a smaller model
            }

            logger.info("Successfully initialized all LLM connectors")
        except Exception as e:
            logger.error(f"Error initializing LLM connectors: {str(e)}")
            raise

    def analyze_market_conditions(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Analyze market conditions using multi-LLM approach"""
        try:
            # Build analysis prompt with JSON instruction
            prompt = f"""Analyze the following market conditions and provide insights in JSON format:
Chain: {data.get('chain')}
Metrics: {json.dumps(data.get('metrics', {}), indent=2)}

You must respond in JSON format with the following structure:
{{
    "trend_analysis": string,
    "risk_level": float,
    "confidence_score": float,
    "recommended_action": string,
    "key_factors": array,
    "position_size_modifier": float
}}
json
"""
            # Get OpenAI analysis
            openai_response = self._get_openai_analysis(prompt)

            # Get Cohere analysis for validation
            cohere_response = self._get_cohere_analysis(prompt)

            # Combine and validate analyses
            return self._combine_analyses(openai_response, cohere_response)

        except Exception as e:
            logger.error(f"Market analysis error: {str(e)}")
            return None

    def _get_openai_analysis(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Get analysis from OpenAI"""
        try:
            response = self.openai_client.chat.completions.create(
                model=self.models['openai'],
                messages=[
                    {
                        "role": "system",
                        "content": "You are a financial analysis expert. Always respond in valid JSON format."
                    },
                    {
                        "role": "user",
                        "content": prompt + "\njson"
                    }
                ],
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"OpenAI analysis error: {str(e)}")
            return None

    def _get_cohere_analysis(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Get analysis from Cohere"""
        try:
            response = self.cohere_client.generate(
                prompt=prompt + "\nRespond only with valid JSON.",
                model=self.models['cohere'],
                max_tokens=500,
                temperature=0.7,
                return_likelihoods='NONE'
            )
            return json.loads(response.generations[0].text)
        except Exception as e:
            logger.error(f"Cohere analysis error: {str(e)}")
            return None

    def _get_huggingface_analysis(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Get analysis from Hugging Face"""
        try:
            response = self.huggingface_client.text_generation(
                prompt + "\nRespond only with valid JSON.",
                model=self.models['huggingface'],
                max_new_tokens=500
            )
            return json.loads(response)
        except Exception as e:
            logger.error(f"Hugging Face analysis error: {str(e)}")
            return None

    def analyze_with_fallback(self, prompt: str, analysis_type: str) -> Optional[Dict[str, Any]]:
        """Analyze with fallback support across multiple LLMs"""
        try:
            # Add JSON instruction to prompt
            json_prompt = f"{prompt}\nRespond in JSON format.\njson"

            # Try OpenAI first
            response = self._get_openai_analysis(json_prompt)
            if response:
                return response

            # Fallback to Cohere
            response = self._get_cohere_analysis(json_prompt)
            if response:
                return response

            # Final fallback to Hugging Face
            return self._get_huggingface_analysis(json_prompt)

        except Exception as e:
            logger.error(f"Analysis fallback error: {str(e)}")
            return None

    def get_status(self) -> Dict[str, Any]:
        """Get LLM service status"""
        try:
            status = {
                'status': 'operational',
                'services': {
                    'openai': 'initializing',
                    'cohere': 'initializing',
                    'huggingface': 'initializing'
                },
                'timestamp': datetime.utcnow().isoformat()
            }

            # Check services with a simple prompt
            test_prompt = "Test response in JSON format.\njson"

            # Check OpenAI status
            try:
                self._get_openai_analysis(test_prompt)
                status['services']['openai'] = 'operational'
            except Exception as e:
                logger.error(f"OpenAI status check failed: {str(e)}")
                status['services']['openai'] = 'error'

            # Check Cohere status
            try:
                self._get_cohere_analysis(test_prompt)
                status['services']['cohere'] = 'operational'
            except Exception as e:
                logger.error(f"Cohere status check failed: {str(e)}")
                status['services']['cohere'] = 'error'

            # Check HuggingFace status
            try:
                self._get_huggingface_analysis(test_prompt)
                status['services']['huggingface'] = 'operational'
            except Exception as e:
                logger.error(f"HuggingFace status check failed: {str(e)}")
                status['services']['huggingface'] = 'error'

            return status

        except Exception as e:
            logger.error(f"Status check error: {str(e)}")
            return {
                'status': 'error',
                'message': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }

    def _combine_analyses(self, openai_analysis: Optional[Dict[str, Any]], 
                       cohere_analysis: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine and validate analyses from multiple LLMs"""
        if not openai_analysis and not cohere_analysis:
            return {
                'trend_analysis': 'neutral',
                'risk_level': 0.5,
                'confidence_score': 0.0,
                'recommended_action': 'insufficient_data',
                'key_factors': ['analysis_unavailable'],
                'position_size_modifier': 0.0
            }

        # Prefer OpenAI analysis if available
        primary = openai_analysis if openai_analysis else cohere_analysis
        if not primary:
            return {
                'trend_analysis': 'neutral',
                'risk_level': 0.5,
                'confidence_score': 0.0,
                'recommended_action': 'hold',
                'key_factors': ['analysis_unavailable'],
                'position_size_modifier': 0.5
            }

        # Validate and normalize values
        return {
            'trend_analysis': primary.get('trend_analysis', 'neutral'),
            'risk_level': max(0.0, min(1.0, primary.get('risk_level', 0.5))),
            'confidence_score': max(0.0, min(1.0, primary.get('confidence_score', 0.5))),
            'recommended_action': primary.get('recommended_action', 'hold'),
            'key_factors': primary.get('key_factors', [])[:5],
            'position_size_modifier': max(0.0, min(1.0, primary.get('position_size_modifier', 0.5)))
        }