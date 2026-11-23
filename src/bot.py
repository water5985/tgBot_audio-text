import asyncio
import sys
import tempfile
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path

from telegram import (BotCommand, BotCommandScopeAllPrivateChats, InlineKeyboardButton,
                      InlineKeyboardMarkup, MenuButtonCommands, Message,
                      ReplyKeyboardRemove, Update)
from telegram.error import TelegramError
from telegram.ext import (Application, CallbackQueryHandler, CommandHandler, ContextTypes,
                          MessageHandler, filters)

from src.processes import ProcessingError
from src.isolation import isolated_transcribe, isolated_transcribe_result
from src.bot_preferences import JobOptions, LANGUAGES
from src.original_results import OriginalResults
from src.settings import Settings
from src.transcription import MAX_AUDIO_SECONDS, transcribe, transcribe_result
from src.translation_isolation import isolated_translate


MAX_FILE_BYTES = 20_000_000
MAX_PENDING_JOBS = 3


async def reply(message: Message, text: str) -> None:
    await message.reply_text(text, parse_mode=None, do_quote=False,
                             disable_web_page_preview=True)


async def reply_chunks(message: Message, text: str) -> None:
    for offset in range(0, len(text), 2000):
        await reply(message, text[offset:offset + 2000])


async def finish_thread(function, *args, **kwargs):
    """Cancellation cannot release a slot or delete input while its worker is alive."""
    task = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
        except Exception:
            break
    if cancelled:
        if not task.cancelled():
            task.exception()
        raise asyncio.CancelledError
    return task.result()


class VoiceBot:
    def __init__(self, settings: Settings, transcriber: Callable | None = None,
                 recognizer: Callable | None = None, translator: Callable | None = None):
        self.settings = settings
        self.transcriber = transcriber or (isolated_transcribe if settings.audio_sandbox else transcribe)
        self.recognizer = recognizer or (isolated_transcribe_result if settings.audio_sandbox else transcribe_result)
        self.translator = translator or isolated_translate
        self.pending_users: set[int] = set()
        self.worker = asyncio.Semaphore(1)
        self.languages: dict[int, str] = {}
        self.modes: dict[int, str] = {}
        self.targets: dict[int, str] = {}
        self.originals = OriginalResults()

    def options(self, user_id: int) -> JobOptions:
        return JobOptions(self.modes.get(user_id, 'transcribe'),
                          self.languages.get(user_id, self.settings.language),
                          self.targets.get(user_id))

    async def shutdown(self, application: Application) -> None:
        self.originals.close()

    async def authorized_message(self, update: Update) -> Message | None:
        message, user = update.message, update.effective_user
        if not message or message.chat.type != 'private' or not user:
            return None
        if not self.settings.allows(user.id):
            await reply(message, 'Access is restricted. Use /id to get your user ID.')
            return None
        return message

    async def identity(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if update.message and update.message.chat.type == "private" and update.effective_user:
            await reply(update.message, f"Your Telegram user ID: {update.effective_user.id}")

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message or update.message.chat.type != "private":
            return
        await update.message.reply_text(
                    f"Send a voice message up to {MAX_AUDIO_SECONDS // 60} minutes and 20 MB. "
                    "Processing can take several minutes. Temporary audio is deleted "
                    "after processing; your messages and replies remain in Telegram. "
                    "Use the Menu commands and buttons to choose settings.",
                    parse_mode=None, do_quote=False, reply_markup=ReplyKeyboardRemove())
        if update.effective_user and self.settings.allows(update.effective_user.id):
            await self.mode_menu(update.message, update.effective_user.id)
        else:
            await reply(update.message, 'Access requires approval; use /id to get your user ID.')

    async def mode_menu(self, message: Message, user_id: int) -> None:
        options = self.options(user_id)
        buttons = [InlineKeyboardButton('Transcribe', callback_data='mode:transcribe')]
        if self.settings.translation_enabled:
            buttons.append(InlineKeyboardButton('Translate', callback_data='mode:translate'))
        await message.reply_text(
            f'Choose a mode. Current: {options.mode}.\n'
            f'Spoken language: {LANGUAGES[options.source]}.\n'
            f'Target language: {LANGUAGES.get(options.target, "not selected")}.\n'
            'Choices reset when the bot restarts.',
            parse_mode=None, do_quote=False, reply_markup=InlineKeyboardMarkup([buttons]))

    async def mode(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        message = await self.authorized_message(update)
        if message is None:
            return
        args = context.args or []
        choice = args[0].lower() if len(args) == 1 else None
        await self.choose_mode(message, update.effective_user.id, choice)

    async def choose_mode(self, message: Message, user_id: int, choice: str | None) -> None:
        if choice not in {'transcribe', 'translate'}:
            await self.mode_menu(message, user_id)
            return
        if choice == 'translate' and not self.settings.translation_enabled:
            await reply(message, 'Translation is not enabled. Transcription is available.')
            return
        self.modes[user_id] = choice
        await reply(message, f'Mode: {choice}. Already accepted recordings keep their settings.')
        await self.language_menu(message, user_id)

    async def language_menu(self, message: Message, user_id: int) -> None:
        current = self.languages.get(user_id, self.settings.language)
        await message.reply_text(
            f"Choose the language of your next recordings. Current: {LANGUAGES[current]}.\n"
            "uk — Українська; en — English; pl — Polski; ru — Русский; auto — detect.\n"
            "Use /language anytime to change it. Choices reset when the bot restarts.",
            parse_mode=None, do_quote=False,
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton(label, callback_data='language:' + code)]
                for code, label in LANGUAGES.items()]))

    async def language(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        message = await self.authorized_message(update)
        if message is None:
            return
        args = context.args or []
        choice = args[0].lower() if len(args) == 1 else None
        await self.choose_language(message, update.effective_user.id, choice)

    async def choose_language(self, message: Message, user_id: int, choice: str | None) -> None:
        if choice not in LANGUAGES:
            await self.language_menu(message, user_id)
            return
        self.languages[user_id] = choice
        await message.reply_text(
            f"Language: {LANGUAGES[choice]}. Send your next voice message. "
            "Already accepted recordings keep their previous setting.",
            parse_mode=None, do_quote=False, reply_markup=ReplyKeyboardRemove())
        if self.modes.get(user_id) == 'translate':
            await self.target_menu(message, user_id)

    async def target_menu(self, message: Message, user_id: int) -> None:
        await message.reply_text(
            f'Choose the target language. Current: {LANGUAGES.get(self.targets.get(user_id), "not selected")}.',
            parse_mode=None, do_quote=False,
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton(label, callback_data='target:' + code)]
                for code, label in LANGUAGES.items() if code != 'auto']))

    async def target(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        message = await self.authorized_message(update)
        if message is None:
            return
        args = context.args or []
        choice = args[0].lower() if len(args) == 1 else None
        await self.choose_target(message, update.effective_user.id, choice)

    async def choose_target(self, message: Message, user_id: int, choice: str | None) -> None:
        if not self.settings.translation_enabled:
            await reply(message, 'Translation is not enabled.')
            return
        if choice not in LANGUAGES or choice == 'auto':
            await self.target_menu(message, user_id)
            return
        self.targets[user_id] = choice
        await reply(message, f'Target language: {LANGUAGES[choice]}. Send a voice message. '
                    'Already accepted recordings keep their settings.')

    async def help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        message = await self.authorized_message(update)
        if message is not None:
            await reply(message, '/mode — Transcribe or Translate\n/language — spoken language or Auto\n'
                        '/target — translation output language\n/id — your user ID\n'
                        'Send a voice message after selecting settings. Ordinary text does not change settings. '
                        'Translations may contain errors. Originals are available for up to 10 minutes '
                        'and may expire earlier when the cache is full or the bot restarts.')

    async def callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        query, user = update.callback_query, update.effective_user
        if query is None:
            return
        await query.answer()
        message = query.message
        if (not isinstance(message, Message) or message.chat.type != 'private' or not user
                or message.chat.id != user.id or not self.settings.allows(user.id)):
            return
        data = query.data
        if not isinstance(data, str) or ':' not in data:
            return
        action, choice = data.split(':', 1)
        if action == 'original':
            original = self.originals.get(choice, user.id, message.chat.id)
            if original is None:
                await reply(message, 'This original has expired or is unavailable. Please resend the recording.')
            else:
                await reply_chunks(message, original)
            return
        handlers = {'mode': self.choose_mode, 'language': self.choose_language, 'target': self.choose_target}
        handler = handlers.get(action)
        if handler is not None:
            await handler(message, user.id, choice)

    async def unsupported(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if (update.message and update.message.chat.type == "private"
                and update.effective_user
                and self.settings.allows(update.effective_user.id)):
            await reply(update.message, "Please send a Telegram voice message. Use /mode, /language and /target for settings.")

    async def voice(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        message = update.message
        user = update.effective_user
        if not message or message.chat.type != "private" or not user or not message.voice:
            return
        if not self.settings.allows(user.id):
            await reply(message, "Access is restricted. Use /id to get your user ID.")
            return
        options = self.options(user.id)
        if options.mode == 'translate' and (not self.settings.translation_enabled or options.target is None):
            await reply(message, 'Choose a target language with /target before sending a recording. '
                        'Translation must be enabled by the operator.')
            return
        voice = message.voice
        duration = voice.duration
        if isinstance(duration, timedelta):
            duration = duration.total_seconds()
        if duration <= 0 or duration > MAX_AUDIO_SECONDS:
            await reply(message, f"Please send a voice message up to {MAX_AUDIO_SECONDS // 60} minutes long.")
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
        context.application.create_task(self.process(message, user.id, options))

    async def process(self, message: Message, user_id: int, options: JobOptions) -> None:
        try:
            async with self.worker:
                result = await self.download_and_transcribe(message, options.source,
                                                           structured=options.mode == 'translate')
                if options.mode == 'translate':
                    await self.deliver_translation(message, user_id, options, result)
                else:
                    await reply_chunks(message, result)
        except Exception:
            print("Voice request failed.", file=sys.stderr)
            try:
                await reply(message, "Could not finish this recording. Please try again later.")
            except (TelegramError, OSError):
                pass
        finally:
            self.pending_users.discard(user_id)

    async def deliver_translation(self, message, user_id, options, result):
        source = result.language
        if source not in LANGUAGES or source == 'auto':
            await reply(message, 'Could not identify a supported spoken language. Choose it with /language. '
                        'Here is the original transcription:')
            await reply_chunks(message, result.text)
            return
        if source == options.target:
            await reply(message, 'Source and target languages are the same; no translation is needed. Original:')
            await reply_chunks(message, result.text)
            return
        try:
            translated = await finish_thread(
                self.translator, result.text, self.settings.translation_model_dir,
                source=source, target=options.target, runtime=self.settings.translation_runtime,
                threads=self.settings.translation_threads, timeout=self.settings.translation_timeout)
            if not isinstance(translated, str) or not translated.strip() or len(translated.encode('utf-8')) > 65_536:
                raise ProcessingError('Invalid translation output.')
        except Exception:
            await reply(message, 'Translation could not be completed. Here is the original transcription:')
            await reply_chunks(message, result.text)
            return
        await reply(message, f'Translation: {LANGUAGES[source]} → {LANGUAGES[options.target]}. '
                    'May contain errors; check important details.')
        await reply_chunks(message, translated)
        key = self.originals.put(user_id, message.chat.id, result.text)
        await message.reply_text('Original available for up to 10 minutes.', parse_mode=None, do_quote=False,
                                 reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton(
                                     'Show original', callback_data='original:' + key)]]))

    async def download_and_transcribe(self, message: Message, language: str | None = None,
                                      structured: bool = False):
        with tempfile.TemporaryDirectory(prefix="voice-") as directory:
            source = Path(directory) / "audio.ogg"
            remote_file = await message.voice.get_file()
            if remote_file.file_size is not None and remote_file.file_size > MAX_FILE_BYTES:
                raise ProcessingError("Recording is too large.")
            await remote_file.download_to_drive(source, read_timeout=60, connect_timeout=10)
            if not 0 < source.stat().st_size <= MAX_FILE_BYTES:
                raise ProcessingError("Recording size is invalid.")
            return await finish_thread(
                self.recognizer if structured else self.transcriber, source, self.settings.model,
                ffmpeg=self.settings.ffmpeg, whisper=self.settings.whisper,
                language=language or self.settings.language, threads=self.settings.threads,
                timeout=self.settings.timeout,
            )


async def handle_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    print("Telegram request failed. Check the connection and bot configuration.", file=sys.stderr)


def build_application(settings: Settings) -> Application:
    service = VoiceBot(settings)
    application = (Application.builder().token(settings.token).concurrent_updates(False)
                   .post_init(register_commands).post_shutdown(service.shutdown).build())
    application.add_handler(CommandHandler("id", service.identity))
    application.add_handler(CommandHandler("start", service.start))
    application.add_handler(CommandHandler("language", service.language))
    application.add_handler(CommandHandler('mode', service.mode))
    application.add_handler(CommandHandler('target', service.target))
    application.add_handler(CommandHandler('help', service.help))
    application.add_handler(CallbackQueryHandler(service.callback))
    application.add_handler(MessageHandler(filters.VOICE, service.voice))
    application.add_handler(MessageHandler(filters.ALL, service.unsupported))
    application.add_error_handler(handle_error)
    return application


async def register_commands(application: Application) -> None:
    await application.bot.set_my_commands([
        BotCommand('start', 'Start and show current settings'),
        BotCommand('mode', 'Choose Transcribe or Translate'),
        BotCommand('language', 'Choose spoken language'),
        BotCommand('target', 'Choose translation output language'),
        BotCommand('help', 'Show help'), BotCommand('id', 'Show your user ID'),
    ], scope=BotCommandScopeAllPrivateChats())
    await application.bot.set_chat_menu_button(menu_button=MenuButtonCommands())
