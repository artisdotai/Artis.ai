"""Ethereum trading module for AI10X platform"""
import os
import logging
from typing import Dict, Any, Optional
from web3 import Web3
from web3.types import TxParams, TxReceipt
from eth_account import Account
from eth_account.signers.local import LocalAccount
import json

logger = logging.getLogger(__name__)

class ETHTrader:
    """
    Handles Ethereum trading operations on Sepolia testnet.
    """
    def __init__(self):
        self.endpoint = f"https://eth-sepolia.g.alchemy.com/v2/{os.environ.get('ALCHEMY_API_KEY')}"
        self.web3 = Web3(Web3.HTTPProvider(self.endpoint))
        
        # Load test wallets
        try:
            with open('test_wallets.json', 'r') as f:
                self.wallets = json.load(f)
                self.eth_wallet = self.wallets.get('eth', {})
        except Exception as e:
            logger.error(f"Error loading wallets: {str(e)}")
            self.wallets = {}
            self.eth_wallet = {}

        # Initialize account if wallet is available
        self.account: Optional[LocalAccount] = None
        if self.eth_wallet:
            try:
                private_key = self.eth_wallet.get('private_key')
                if private_key:
                    self.account = Account.from_key(private_key)
                    logger.info(f"Initialized ETH account: {self.account.address}")
            except Exception as e:
                logger.error(f"Error initializing ETH account: {str(e)}")

        self.gas_price_multiplier = 1.1  # 10% buffer for gas price
        self.max_retries = 3
        self.retry_delay = 2

    def validate_connection(self) -> bool:
        """Verify connection to Ethereum network"""
        try:
            return self.web3.is_connected() and bool(self.account)
        except Exception as e:
            logger.error(f"Connection validation failed: {str(e)}")
            return False

    def get_balance(self) -> float:
        """Get ETH balance for trading account"""
        try:
            if not self.account:
                return 0.0
            balance_wei = self.web3.eth.get_balance(self.account.address)
            return float(self.web3.from_wei(balance_wei, 'ether'))
        except Exception as e:
            logger.error(f"Error getting balance: {str(e)}")
            return 0.0

    def estimate_gas_price(self) -> int:
        """Get current gas price with buffer"""
        try:
            base_fee = self.web3.eth.gas_price
            return int(base_fee * self.gas_price_multiplier)
        except Exception as e:
            logger.error(f"Error estimating gas price: {str(e)}")
            return 0

    def send_eth(self, to_address: str, amount_eth: float) -> Dict[str, Any]:
        """
        Send ETH to specified address
        Returns transaction details or error information
        """
        if not self.validate_connection():
            return {'success': False, 'error': 'Not connected to network'}

        try:
            # Convert ETH to Wei
            amount_wei = self.web3.to_wei(amount_eth, 'ether')
            
            # Check balance
            balance = self.get_balance()
            if balance < amount_eth:
                return {
                    'success': False, 
                    'error': f'Insufficient balance. Have {balance} ETH, need {amount_eth} ETH'
                }

            # Prepare transaction
            gas_price = self.estimate_gas_price()
            nonce = self.web3.eth.get_transaction_count(self.account.address)
            
            tx_params: TxParams = {
                'nonce': nonce,
                'to': to_address,
                'value': amount_wei,
                'gas': 21000,  # Standard ETH transfer gas
                'gasPrice': gas_price,
                'chainId': 11155111  # Sepolia chain ID
            }

            # Sign and send transaction
            signed_tx = self.account.sign_transaction(tx_params)
            tx_hash = self.web3.eth.send_raw_transaction(signed_tx.rawTransaction)
            
            # Wait for transaction receipt
            tx_receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash)
            
            return {
                'success': True,
                'transaction_hash': tx_receipt.get('transactionHash', '').hex(),
                'block_number': tx_receipt.get('blockNumber'),
                'gas_used': tx_receipt.get('gasUsed'),
                'status': tx_receipt.get('status')
            }

        except Exception as e:
            logger.error(f"Error sending ETH: {str(e)}")
            return {'success': False, 'error': str(e)}

    def verify_transaction(self, tx_hash: str) -> Dict[str, Any]:
        """Verify transaction status"""
        try:
            tx_receipt = self.web3.eth.get_transaction_receipt(tx_hash)
            return {
                'success': bool(tx_receipt.get('status')),
                'block_number': tx_receipt.get('blockNumber'),
                'gas_used': tx_receipt.get('gasUsed'),
                'confirmations': self.web3.eth.block_number - tx_receipt.get('blockNumber')
            }
        except Exception as e:
            logger.error(f"Error verifying transaction: {str(e)}")
            return {'success': False, 'error': str(e)}

    def get_transaction_history(self, limit: int = 10) -> list:
        """Get recent transactions for the account"""
        try:
            if not self.account:
                return []

            transactions = []
            current_block = self.web3.eth.block_number
            
            # Get the last 'limit' number of transactions
            for i in range(limit):
                try:
                    tx = self.web3.eth.get_transaction_by_block(
                        current_block - i,
                        0  # First transaction in block
                    )
                    if tx and (tx.get('from') == self.account.address or 
                              tx.get('to') == self.account.address):
                        transactions.append({
                            'hash': tx.get('hash', '').hex(),
                            'from': tx.get('from'),
                            'to': tx.get('to'),
                            'value': float(self.web3.from_wei(tx.get('value', 0), 'ether')),
                            'block_number': tx.get('blockNumber')
                        })
                except Exception as e:
                    logger.debug(f"No transaction in block {current_block - i}: {str(e)}")
                    continue

            return transactions

        except Exception as e:
            logger.error(f"Error getting transaction history: {str(e)}")
            return []
