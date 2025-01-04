"""Test PumpFun analyzer functionality with Spydefi integration"""
import logging
from flask import current_app
from app import app
from pumpfun_analyzer import PumpFunAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_early_launches():
    """Test early launches detection"""
    analyzer = PumpFunAnalyzer()

    # Get early launches with test data
    launches = analyzer.get_early_launches(chain='solana')

    print("\n=== Early Token Launches ===")
    for launch in launches:
        print(f"\nToken: {launch['token_address']}")
        print(f"Launch Time: {launch['launch_time']}")
        print(f"Liquidity: ${launch['liquidity']:,.2f}")
        print(f"Volume 24h: ${launch['volume_24h']:,.2f}")
        print(f"Holder Count: {launch['holder_count']}")
        if 'potential_score' in launch:
            print(f"Potential Score: {launch['potential_score']:.1f}/10")

def test_token_analysis():
    """Test token analysis capabilities"""
    analyzer = PumpFunAnalyzer()

    # Test with mock token data
    test_token = {
        'address': 'SOLTEST1' * 8,
        'price': 0.00001,
        'volume24h': 15000,
        'liquidity': 75000,
        'holders': 150,
        'top_holder_percentage': 8.5,
        'is_contract_verified': True,
        'is_mint_enabled': False,
        'price_history': [
            {'price': 0.000008},
            {'price': 0.00001}
        ]
    }

    success, analysis = analyzer.analyze_token_potential(test_token)

    print("\n=== Token Analysis ===")
    if success:
        print(f"\nToken Metrics:")
        print(f"Price: ${analysis['metrics']['price']:,.8f}")
        print(f"Volume 24h: ${analysis['metrics']['volume_24h']:,.2f}")
        print(f"Liquidity: ${analysis['metrics']['liquidity']:,.2f}")
        print(f"Holder Count: {analysis['metrics']['holder_count']}")
        print(f"Price Change: {analysis['metrics']['price_change']:.1f}%")

        print(f"\nRisk Assessment:")
        print(f"Risk Level: {analysis['risk_assessment']['risk_level']}")
        print(f"Risk Factors: {', '.join(analysis['risk_assessment']['risk_factors'])}")
        print(f"Safety Score: {analysis['risk_assessment']['safety_score']:.1f}/100")

        print(f"\nPotential Score: {analysis['potential_score']:.1f}/10")
    else:
        print(f"Analysis failed: {analysis.get('error', 'Unknown error')}")

def test_social_trading():
    """Test social trading metrics functionality"""
    analyzer = PumpFunAnalyzer()

    print("\n=== Social Trading Metrics ===")
    test_token = 'SOLTEST1' * 8

    with app.app_context():
        metrics = analyzer.get_social_trading_metrics(test_token)

        if 'error' in metrics:
            print(f"\nError getting metrics: {metrics['error']}")
            return

        print(f"\nToken: {metrics['token_address']}")
        print(f"Social Score: {metrics['social_score']:.1f}")
        print(f"Total Trades: {metrics['total_trades']:,}")
        print(f"Successful Trades: {metrics['successful_trades']:,}")
        print(f"Total Volume: ${metrics['total_volume']:,.2f}")
        print(f"ROI: {metrics['roi_percentage']:.1f}%")
        print(f"Trader Rank: #{metrics['trader_rank']:,}")
        print(f"Sentiment Score: {metrics['sentiment_score']:.2f}")
        print(f"Engagement Rate: {metrics['engagement_rate']:.2%}")
        print(f"Trend Direction: {metrics['trend_direction']}")

def test_leaderboard():
    """Test leaderboard functionality"""
    analyzer = PumpFunAnalyzer()

    print("\n=== Top Traders Leaderboard ===")
    with app.app_context():
        top_traders = analyzer.get_top_traders(limit=5)

        for i, trader in enumerate(top_traders, 1):
            print(f"\n{i}. {trader['username']}")
            print(f"   Reputation Score: {trader['reputation_score']:.1f}")
            print(f"   Win Rate: {trader['win_rate']:.1f}%")
            print(f"   Total PnL: ${trader['total_pnl']:,.2f}")
            print(f"   Level: {trader['level']} ({trader['experience_points']} XP)")
            print(f"   Badges: {', '.join(trader['badges'])}")
            print(f"   Streak: {trader['streak_days']} days")

def main():
    """Run PumpFun analyzer tests"""
    print("\nTesting PumpFun Analyzer...")

    with app.app_context():
        test_early_launches()
        test_token_analysis()
        test_social_trading()
        test_leaderboard()

if __name__ == "__main__":
    main()