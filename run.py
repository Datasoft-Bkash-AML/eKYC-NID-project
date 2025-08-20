import logging
import uvicorn
from main import app
import os
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def serve():
    """Serving the web application."""
    try: 
        port = int(os.environ.get("PORT", 8080))
        uvicorn.run(app,
                host="0.0.0.0",
                port=port,
                reload=False,
                access_log=True, 
                log_level="debug"
            )
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        input("Press Enter to exit...")

if __name__ == "__main__":
    serve()