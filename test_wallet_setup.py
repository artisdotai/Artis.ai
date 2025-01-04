"""Test Solana wallet and connection functionality"""
import os
import logging
from solana.rpc.api import Client as SolanaClient
from solders.keypair import Keypair
import base58
from typing import Dict, Any, Optional
from web3 import Web3
from eth_account import Account
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_solana_connection():
    """Test Solana network connection and configuration"""
    try:
        # Use provided Solana API key if available
        if os.environ.get('SOLANA_API_KEY'):
            endpoint = f"https://api.devnet.solana.com/{os.environ.get('SOLANA_API_KEY')}"
        else:
            endpoint = "https://api.devnet.solana.com"

        client = SolanaClient(endpoint)

        # Test connection using get_version
        version = client.get_version()
        logger.info(f"Successfully connected to Solana network")
        logger.info(f"Network version: {version.value}")

        # Get recent blockhash to further verify connection
        recent_blockhash = client.get_latest_blockhash()
        logger.info(f"Recent blockhash obtained: {recent_blockhash.value.blockhash}")

        return True, client

    except Exception as e:
        logger.error(f"Error connecting to Solana network: {str(e)}")
        return False, None

class TestWalletManager:
    def __init__(self):
        self.rpc_endpoints = {
            'eth': f"https://eth-sepolia.g.alchemy.com/v2/{os.environ.get('ALCHEMY_API_KEY')}",
            #'solana': 'https://api.devnet.solana.com' # Removed - use test_solana_connection instead
        }

        # Maximum retry attempts for connections
        self.max_retries = 3
        self.retry_delay = 2  # seconds

        self.web3_providers = {}
        self.test_wallets = {}
        self.initialize_providers()

    def initialize_providers(self):
        """Initialize Web3 providers with retry logic"""
        for chain, endpoint in self.rpc_endpoints.items():
            if chain == 'eth':
                if os.environ.get('ALCHEMY_API_KEY'):
                    try:
                        web3 = Web3(Web3.HTTPProvider(endpoint))
                        # Test connection
                        if web3.is_connected():
                            self.web3_providers[chain] = web3
                            logger.info(f"Successfully connected to {chain.upper()} network")
                        else:
                            logger.warning(f"Could not establish connection to {chain.upper()} network")
                    except Exception as e:
                        logger.error(f"Error initializing {chain.upper()} provider: {str(e)}")
                else:
                    logger.info("Skipping ETH initialization - Alchemy API key not provided")


    def generate_eth_wallet(self) -> Dict[str, str]:
        """Generate Ethereum-compatible wallet for ETH"""
        try:
            account = Account.create()
            logger.info("Generated ETH-compatible wallet")
            return {
                'address': account.address,
                'private_key': account.key.hex(),
                'chain': 'eth'
            }
        except Exception as e:
            logger.error(f"Error generating ETH wallet: {str(e)}")
            raise

    def generate_solana_wallet(self) -> Dict[str, str]:
        """Generate Solana wallet with improved error handling"""
        try:
            keypair = Keypair()
            secret_bytes = bytes(keypair)
            secret_key = secret_bytes[:32]

            wallet_info = {
                'address': str(keypair.pubkey()),
                'private_key': base58.b58encode(secret_key).decode('ascii'),
                'chain': 'solana'
            }
            logger.info("Generated Solana wallet")
            return wallet_info
        except Exception as e:
            logger.error(f"Error generating Solana wallet: {str(e)}")
            raise

    def setup_test_wallets(self) -> Dict[str, Dict[str, str]]:
        """Setup test wallets for ETH and Solana"""
        logger.info("Generating test wallets for ETH and Solana...")

        try:
            # Generate ETH wallet
            self.test_wallets['eth'] = self.generate_eth_wallet()

            # Generate Solana wallet
            self.test_wallets['solana'] = self.generate_solana_wallet()

            logger.info("Successfully generated all test wallets")
            self._save_wallets()
            return self.test_wallets
        except Exception as e:
            logger.error(f"Error setting up test wallets: {str(e)}")
            raise

    def _save_wallets(self):
        """Save wallet information to file with error handling"""
        try:
            with open('test_wallets.json', 'w') as f:
                json.dump(self.test_wallets, f, indent=2)
            logger.info("Saved wallet information to test_wallets.json")
        except Exception as e:
            logger.error(f"Error saving wallet information: {str(e)}")

    def verify_balances(self, solana_client):
        """Check balances across all chains with improved error handling"""
        print("\n=== Checking Balances ===")

        for chain, wallet in self.test_wallets.items():
            try:
                if chain == 'solana':
                    try:
                        pubkey = Pubkey.from_string(wallet['address'])
                        balance_response = solana_client.get_balance(pubkey)

                        # Handle the response properly based on Solana client response type
                        if hasattr(balance_response, 'value'):
                            balance = balance_response.value
                            print(f"Solana balance: {balance / 1e9:.9f} SOL")
                        else:
                            print("Solana balance: Error - Invalid response format")

                    except Exception as e:
                        logger.error(f"Error checking Solana balance: {str(e)}")
                        print("Solana balance: Error checking balance")
                elif chain == 'eth' and os.environ.get('ALCHEMY_API_KEY'):
                    try:
                        if self.web3_providers.get('eth'):
                            web3 = self.web3_providers['eth']
                            balance_wei = web3.eth.get_balance(wallet['address'])
                            balance_eth = web3.from_wei(balance_wei, 'ether')
                            print(f"ETH balance: {balance_eth:.9f} ETH")
                        else:
                            print("ETH balance check skipped - Provider not initialized")
                    except Exception as e:
                        logger.error(f"Error checking ETH balance: {str(e)}")
                        print("ETH balance: Error checking balance")
                else:
                    print("ETH balance check skipped - Alchemy API key not configured")

            except Exception as e:
                logger.error(f"Error checking {chain} balance: {str(e)}")
                print(f"{chain.upper()} balance: Error checking balance")

def main():
    """Main function to setup and display test wallet information"""
    try:
        wallet_manager = TestWalletManager()
        wallets = wallet_manager.setup_test_wallets()

        print("\n=== Test Wallet Information ===")
        for chain, wallet in wallets.items():
            print(f"\n{chain.upper()} Wallet:")
            print(f"Address: {wallet['address']}")
            # Only show first/last 4 chars of private key for security
            pk = wallet['private_key']
            print(f"Private Key: {pk[:4]}...{pk[-4:]}")

        print("\nPlease fund these wallets using the following faucets:")
        print("ETH (Sepolia): https://sepoliafaucet.com/")
        print("Solana (Devnet): https://solfaucet.com")

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    import sys
    try:
        # Test Solana connection first
        print("\n=== Testing Solana Connection ===")
        success, client = test_solana_connection()
        print(f"Connection successful: {success}")

        if success:
            print("Solana network is properly configured and accessible")
            wallet_manager = TestWalletManager()
            if len(sys.argv) > 1 and sys.argv[1] == '--check-balance':
                wallet_manager.setup_test_wallets()
                wallet_manager.verify_balances(client)
            else:
                main()
        else:
            print("Failed to connect to Solana network - please check configuration")


    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)