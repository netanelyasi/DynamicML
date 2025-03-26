# app/logger.py

import logging
from pathlib import Path

# הגדרת הלוגר עם פורמט ורמת לוגינג
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(Path(__file__).stem)
