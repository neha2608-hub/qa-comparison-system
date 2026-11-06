"""Text-to-Speech with gTTS (needs an internet connection)."""
import io
import config
from utils.model_loader import classify_error


class TTSError(Exception):
    """Raised when audio cannot be generated."""


def text_to_speech(text: str, lang: str = config.TTS_LANGUAGE) -> bytes:
    """Convert text to MP3 bytes."""
    if not text or not text.strip():
        raise TTSError("There is no answer text to read aloud.")
    try:
        from gtts import gTTS
        buf = io.BytesIO()
        gTTS(text=text.strip(), lang=lang).write_to_fp(buf)
        data = buf.getvalue()
    except Exception as exc:
        if classify_error(exc) == "network":
            raise TTSError("Could not reach the Google Text-to-Speech service. gTTS needs an "
                           "internet connection each time audio is generated. Check your "
                           "connection/VPN/firewall and try again.") from exc
        raise TTSError(f"Audio generation failed: {exc}") from exc
    if not data:
        raise TTSError("gTTS returned empty audio.")
    return data
