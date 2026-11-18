import asyncio
import sys
import tempfile
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path

from telegram import Message, ReplyKeyboardMarkup, ReplyKeyboardRemove, Update
from telegram.error import TelegramError
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from src.processes import ProcessingError
from src.settings import Settings
from src.transcription import transcribe


MAX_FILE_BYTES = 20_000_000
MAX_PENDING_JOBS = 3
LANGUAGES = {'uk': 'Українська', 'en': 'English', 'pl': 'Polski',
             'ru': 'Русский', 'auto': 'Automatic detection'}


async def reply(message: Message, text: str) -> None:
    await message.reply_text(text, parse_mode=None, do_quote=False,
                             disable_web_page_preview=True)


class VoiceBot:
    def __init__(self, settings: Settings, transcriber: Callable = transcribe):
        self.settings = settings
        self.transcriber = transcriber
        self.pending_users: set[int] = set()
        self.worker = asyncio.Semaphore(1)
        self.languages: dict[int, str] = {}

    async def identity(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if update.message and update.message.chat.type == "private" and update.effective_user:
            await reply(update.message, f"Your Telegram user ID: {update.effective_user.id}")

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message or update.message.chat.type != "private":
            return
        await reply(update.message,
                    "Send a voice message up to five minutes and 20 MB. "
                    "Access requires approval; use /id to get your user ID. "
                    "Processing can take several minutes. Temporary audio is deleted "
                    "after processing; your messages and replies remain in Telegram.")
        if update.effective_user and update.effective_user.id in self.settings.allowed_user_ids:
            await self.language_menu(update.message, update.effective_user.id)

    async def language_menu(self, message: Message, user_id: int) -> None:
        current = self.languages.get(user_id, self.settings.language)
        await message.reply_text(
            f"Choose the language of your next recordings. Current: {LANGUAGES[current]}.\n"
            "uk — Українська; en — English; pl — Polski; ru — Русский; auto — detect.\n"
            "Use /language anytime to change it. Choices reset when the bot restarts.",
            parse_mode=None, do_quote=False,
            reply_markup=ReplyKeyboardMarkup(
                [['/language uk', '/language en'], ['/language pl', '/language ru'],
                 ['/language auto']], resize_keyboard=True, one_time_keyboard=True))

    async def language(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        message, user = update.message, update.effective_user
        if not message or message.chat.type != 'private' or not user:
            return
        if user.id not in self.settings.allowed_user_ids:
            await reply(message, 'Access is restricted. Use /id to get your user ID.')
            return
        args = context.args or []
        choice = args[0].lower() if len(args) == 1 else None
        if choice not in LANGUAGES:
            await self.language_menu(message, user.id)
            return
        self.languages[user.id] = choice
        await message.reply_text(
            f"Language: {LANGUAGES[choice]}. Send your next voice message. "
            "Already accepted recordings keep their previous setting.",
            parse_mode=None, do_quote=False, reply_markup=ReplyKeyboardRemove())

    async def unsupported(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if (update.message and update.message.chat.type == "private"
                and update.effective_user
                and update.effective_user.id in self.settings.allowed_user_ids):
            await reply(update.message, "Please send a Telegram voice message.")

    async def voice(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        message = update.message
        user = update.effective_user
        if not message or message.chat.type != "private" or not user or not message.voice:
            return
        if user.id not in self.settings.allowed_user_ids:
            await reply(message, "Access is restricted. Use /id to get your user ID.")
            return
        voice = message.voice
        duration = voice.duration
        if isinstance(duration, timedelta):
            duration = duration.total_seconds()
        if duration <= 0 or duration > 300:
            await reply(message, "Please send a voice message up to five minutes long.")
            return
        if voice.file_size is None or not 0 < voice.file_size <= MAX_FILE_BYTES:
            await reply(message, "The recording must have a known size of at most 20 MB.")
            return
        if user.id in self.pending_users:
            await reply(message, "Please wait for your previous recording to finish.")
            return
        if len(self.pending_users) >= MAX_PENDING_JOBS:
            await reply(message, "The queue is full. Please try again later.")
            return
        if not context.application.running:
            await reply(message, "The bot is stopping. Please resend your recording later.")
            return
        self.pending_users.add(user.id)
        language = self.languages.get(user.id, self.settings.language)
        try:
            await reply(message, "Recording accepted. Processing may take several minutes.")
        except asyncio.CancelledError:
            self.pending_users.discard(user.id)
            raise
        except (TelegramError, OSError):
            self.pending_users.discard(user.id)
            return
        if not context.application.running:
            self.pending_users.discard(user.id)
            await reply(message, "The bot is stopping. Please resend your recording later.")
            return
        context.application.create_task(self.process(message, user.id, language))

    async def process(self, message: Message, user_id: int, language: str) -> None:
        try:
            async with self.worker:
                text = await self.download_and_transcribe(message, language)
                for offset in range(0, len(text), 2000):
                    await reply(message, text[offset:offset + 2000])
        except Exception:
            print("Voice request failed.", file=sys.stderr)
            try:
                await reply(message, "Could not finish this recording. Please try again later.")
            except (TelegramError, OSError):
                pass
        finally:
            self.pending_users.discard(user_id)

    async def download_and_transcribe(self, message: Message, language: str | None = None) -> str:
        with tempfile.TemporaryDirectory(prefix="voice-") as directory:
            source = Path(directory) / "audio.ogg"
            remote_file = await message.voice.get_file()
            if remote_file.file_size is not None and remote_file.file_size > MAX_FILE_BYTES:
                raise ProcessingError("Recording is too large.")
            await remote_file.download_to_drive(source, read_timeout=60, connect_timeout=10)
            if not 0 < source.stat().st_size <= MAX_FILE_BYTES:
                raise ProcessingError("Recording size is invalid.")
            task = asyncio.create_task(asyncio.to_thread(
                self.transcriber, source, self.settings.model,
                ffmpeg=self.settings.ffmpeg, whisper=self.settings.whisper,
                language=language or self.settings.language, threads=self.settings.threads,
                timeout=self.settings.timeout,
            ))
            try:
                return await asyncio.shield(task)
            except asyncio.CancelledError:
                try:
                    await task
                finally:
                    raise


async def handle_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    print("Telegram request failed. Check the connection and bot configuration.", file=sys.stderr)


def build_application(settings: Settings) -> Application:
    service = VoiceBot(settings)
    application = Application.builder().token(settings.token).concurrent_updates(False).build()
    application.add_handler(CommandHandler("id", service.identity))
    application.add_handler(CommandHandler("start", service.start))
    application.add_handler(CommandHandler("language", service.language))
    application.add_handler(MessageHandler(filters.VOICE, service.voice))
    application.add_handler(MessageHandler(filters.ALL, service.unsupported))
    application.add_error_handler(handle_error)
    return application
