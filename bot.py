import asyncio
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from telegram.error import TelegramError

from src.bot import build_application
from src.settings import ConfigurationError, Settings


def main() -> int:
    logging.disable(logging.CRITICAL)
    os.environ.setdefault("PTB_TIMEDELTA", "1")
    load_dotenv(Path(__file__).with_name(".env"), interpolate=False)
    try:
        settings = Settings.from_env(os.environ)
        settings.validate_runtime()
    except ConfigurationError as error:
        print(str(error), file=sys.stderr)
        return 1
    asyncio.set_event_loop(asyncio.new_event_loop())
    try:
        application = build_application(settings)
        application.run_polling(allowed_updates=["message"], bootstrap_retries=0)
    except (TelegramError, OSError):
        print("Could not run the bot. Check BOT_TOKEN and the network connection.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
