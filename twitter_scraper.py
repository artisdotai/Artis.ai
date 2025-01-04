"""Twitter scraping module for memecoin analysis"""
import logging
import os
from typing import Dict, List, Optional
import tweepy
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class TwitterScraper:
    """Twitter data collection for memecoin analysis"""
    
    def __init__(self):
        """Initialize Twitter API client"""
        try:
            self.api = tweepy.Client(
                bearer_token=os.environ.get("TWITTER_BEARER_TOKEN"),
                wait_on_rate_limit=True
            )
            self.initialized = True
            logger.info("Twitter scraper initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Twitter scraper: {str(e)}")
            self.initialized = False
            raise

        # Default influencer list
        self.influencers = {
            "solana_experts": [
                "SolanaLegend",
                "SolanaFloor",
                "SolanaNews",
                "SolanaDaily"
            ]
        }
        
    def scan_for_memecoins(self, hours: int = 24) -> List[Dict]:
        """Scan Twitter for memecoin mentions"""
        try:
            results = []
            since_time = datetime.utcnow() - timedelta(hours=hours)
            
            # Search tweets from tracked influencers
            for influencer in self.influencers['solana_experts']:
                # Search user's tweets
                tweets = self.api.get_users_tweets(
                    self.get_user_id(influencer),
                    start_time=since_time,
                    max_results=100,
                    tweet_fields=['created_at', 'public_metrics']
                )
                
                if not tweets.data:
                    continue
                    
                for tweet in tweets.data:
                    # Extract potential token mentions
                    mentions = self._extract_token_mentions(tweet.text)
                    if mentions:
                        results.append({
                            'tweet_id': tweet.id,
                            'author': influencer,
                            'text': tweet.text,
                            'created_at': tweet.created_at,
                            'token_mentions': mentions,
                            'metrics': tweet.public_metrics
                        })
            
            return results
            
        except Exception as e:
            logger.error(f"Error scanning Twitter: {str(e)}")
            return []

    def get_user_id(self, username: str) -> Optional[int]:
        """Get Twitter user ID from username"""
        try:
            user = self.api.get_user(username=username)
            return user.data.id if user.data else None
        except Exception as e:
            logger.error(f"Error getting user ID for {username}: {str(e)}")
            return None

    def _extract_token_mentions(self, text: str) -> List[str]:
        """Extract potential token mentions from tweet text"""
        import re
        
        # Common patterns for token mentions
        patterns = [
            r'\$([A-Za-z0-9]+)',  # $TOKEN format
            r'([A-Z]{3,10}(?=\s|$))',  # UPPERCASE tokens
        ]
        
        mentions = []
        for pattern in patterns:
            matches = re.finditer(pattern, text)
            mentions.extend([m.group(1) for m in matches])
        
        return list(set(mentions))  # Remove duplicates

    def update_influencer_list(self, new_influencers: List[str], category: str = 'solana_experts') -> bool:
        """Update the list of tracked influencers"""
        try:
            # Validate usernames exist
            valid_users = []
            for username in new_influencers:
                if self.get_user_id(username):
                    valid_users.append(username)
                else:
                    logger.warning(f"Invalid Twitter username: {username}")
            
            if valid_users:
                self.influencers[category] = valid_users
                logger.info(f"Updated influencer list for {category}: {valid_users}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error updating influencer list: {str(e)}")
            return False

    def get_influencer_metrics(self) -> Dict:
        """Get metrics about tracked influencers"""
        try:
            metrics = {
                'total_influencers': sum(len(group) for group in self.influencers.values()),
                'categories': {
                    category: {
                        'count': len(influencers),
                        'usernames': influencers
                    }
                    for category, influencers in self.influencers.items()
                }
            }
            return metrics
        except Exception as e:
            logger.error(f"Error getting influencer metrics: {str(e)}")
            return {}
