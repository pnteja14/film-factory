import logging
import sys
from pathlib import Path
from datetime import datetime

def setup_logger(name: str, level=logging.INFO) -> logging.Logger:
    """Configure a logger with file and console handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if logger.hasHandlers():
        return logger
    
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / f"film_factory_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(level)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

class AgentLogger:
    """Wrapper for agent-specific logging."""
    
    def __init__(self, agent_name: str):
        self.logger = setup_logger(f"Agent.{agent_name}")
        self.agent_name = agent_name
    
    def info(self, message: str):
        self.logger.info(f"[{self.agent_name}] {message}")
    
    def error(self, message: str, exception: Exception = None):
        if exception:
            self.logger.error(f"[{self.agent_name}] {message}", exc_info=exception)
        else:
            self.logger.error(f"[{self.agent_name}] {message}")
    
    def warning(self, message: str):
        self.logger.warning(f"[{self.agent_name}] {message}")
    
    def debug(self, message: str):
        self.logger.debug(f"[{self.agent_name}] {message}")