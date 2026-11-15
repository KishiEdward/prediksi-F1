"""Entry point server: python -m app.main"""
import uvicorn

from app.config import get_settings


def main() -> None:
    s = get_settings()
    uvicorn.run("app.api.main:app", host="0.0.0.0", port=s.api_port)


if __name__ == "__main__":
    main()
