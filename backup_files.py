"""Script to backup current implementation files"""
import os
import shutil
from datetime import datetime

def create_backup():
    # Create backup directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = f"current_implementation_backup_{timestamp}"
    os.makedirs(backup_dir, exist_ok=True)
    
    # Create subdirectories
    os.makedirs(f"{backup_dir}/static/js", exist_ok=True)
    os.makedirs(f"{backup_dir}/templates", exist_ok=True)
    
    # List of core files to backup
    core_files = [
        "app.py", "main.py", "routes.py", "models.py", "config.py",
        "extensions.py", "database.py"
    ]
    
    # List of analysis module files
    analysis_files = [
        "pumpfun_analyzer.py", "gmgn_analyzer.py", "kol_analyzer.py",
        "market_analysis.py", "sentiment_analyzer.py", "token_analysis.py",
        "technical_analysis.py"
    ]
    
    # List of monitoring module files
    monitoring_files = [
        "solana_monitor.py", "eth_monitor.py", "autonomous_monitor.py",
        "telegram_monitor.py", "solana_token_monitor.py", "monitoring_base.py"
    ]
    
    # List of trading and system files
    trading_files = [
        "trade_executor.py", "eth_trader.py", "profit_manager.py",
        "risk_manager.py", "autonomous_system.py", "autonomous_controller.py",
        "autonomous_optimizer.py"
    ]
    
    # List of integration files
    integration_files = [
        "api_manager.py", "llm_connector.py", "spydefi_connector.py",
        "spyfu_connector.py", "telegram_analyzer.py", "twitter_scraper.py"
    ]
    
    # Combine all files
    all_files = (core_files + analysis_files + monitoring_files + 
                trading_files + integration_files)
    
    # Copy files
    for file in all_files:
        if os.path.exists(file):
            shutil.copy2(file, f"{backup_dir}/{file}")
    
    # Copy static files
    if os.path.exists("static/js/dashboard.js"):
        shutil.copy2("static/js/dashboard.js", f"{backup_dir}/static/js/")
    
    # Copy template files
    template_dir = "templates"
    if os.path.exists(template_dir):
        for file in os.listdir(template_dir):
            if file.endswith('.html'):
                shutil.copy2(f"{template_dir}/{file}", 
                           f"{backup_dir}/templates/{file}")
    
    print(f"Backup completed in directory: {backup_dir}")
    return backup_dir

if __name__ == "__main__":
    backup_dir = create_backup()
