from datetime import datetime
import json
import logging
import os
from typing import Any, List, Union

LOG_DIR = "logs"

def ensure_logs_folder():
    log_folder = "logs"
    os.makedirs(log_folder, exist_ok=True)
    return log_folder

def get_log_file_path():
    today = datetime.today().strftime("%Y-%m-%d")
    folder = ensure_logs_folder()
    return os.path.join(folder, f"{today}.log")

def get_logger():
    logger = logging.getLogger("ec_logger")
    logger.setLevel(logging.INFO)

    # Avoid adding handlers multiple times
    if not logger.handlers:
        file_handler = logging.FileHandler(get_log_file_path())
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

def log_json(note: str, data: Union[dict, list, List[Any]]):
    logger = get_logger()
    try:
        if isinstance(data, list):
            formatted = "\n".join(
                f"[{i+1}] {json.dumps(entry, ensure_ascii=False, indent=2)}"
                for i, entry in enumerate(data)
            )
        else:
            formatted = json.dumps(data, ensure_ascii=False, indent=2)

        logger.info(f"{note}:\n{formatted}")

    except Exception as e:
        logger.error(f"❌ Failed to log JSON: {e}")