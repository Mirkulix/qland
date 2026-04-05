"""Entry point: python -m image_search"""

import uvicorn
from image_search.config import Config


def main():
    config = Config.from_env()
    uvicorn.run(
        "image_search.api.app:app",
        host=config.host,
        port=config.port,
        reload=config.debug,
    )


if __name__ == "__main__":
    main()
