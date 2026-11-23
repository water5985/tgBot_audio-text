# Telegram Voice-to-Text Bot

A self-hosted Telegram bot that turns voice messages into text using whisper.cpp, with optional translation powered by M2M100. Speech recognition and translation run locally on the host machine.

## What it does

- Transcribes Telegram voice messages in private chats.
- Automatically detects the spoken language, or lets users select English, Polish, Ukrainian, or Russian.
- Optionally translates between these four languages.
- Provides a “Show original” button to view the transcript after translation.
- Supports recordings up to 10 minutes and 20 MB.
- Lets users choose their mode and languages through commands and buttons.
- Supports restricted access through a user allowlist or access for everyone.

Temporary audio files are deleted after processing. Messages and replies remain in Telegram, and user preferences reset when the bot restarts.

The project also includes a command-line tool for transcribing local audio files.
