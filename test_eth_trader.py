"""Test script for ETH trading functionality"""
import logging
from eth_trader import ETHTrader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Test ETH trader functionality"""
    try:
        trader = ETHTrader()
        
        # Test connection
        print("\n=== Testing ETH Trader ===")
        print(f"Connected to network: {trader.validate_connection()}")
        
        # Check balance
        balance = trader.get_balance()
        print(f"Current balance: {balance:.6f} ETH")
        
        # Get gas price
        gas_price = trader.estimate_gas_price()
        print(f"Current gas price: {gas_price} Wei")
        
        # Get transaction history
        print("\nRecent transactions:")
        transactions = trader.get_transaction_history(5)
        for tx in transactions:
            print(f"Hash: {tx['hash']}")
            print(f"From: {tx['from']}")
            print(f"To: {tx['to']}")
            print(f"Value: {tx['value']} ETH")
            print("---")

    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
