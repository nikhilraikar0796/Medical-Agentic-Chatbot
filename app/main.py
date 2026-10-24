
import base64
import secrets
import logging

from pathlib import Path
from uuid import uuid4


logger = logging.getLogger(__name__)


from fastapi import (
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    UploadFile,
)
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from groq import Groq

from app.config import settings
from app.agent import graph


ROOT = Path(__file__).resolve().parent.parent

app = FastAPI(
    title="Medical Agent App",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in settings.allowed_origins.split(",")
        if origin.strip()
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type", "X-Conversation-ID"],
)

app.mount(
    "/static",
    StaticFiles(directory=ROOT / "static"),
    name="static",
)


# --------------------------------------------------
# Request schema
# --------------------------------------------------

class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    history: list[dict[str, str]] = Field(
        default_factory=list,
        max_length=10,
    )
    image_data: str | None = None

# --------------------------------------------------
# Groq client
# --------------------------------------------------

def get_groq_client():
    if not settings.groq_api_key:
        raise HTTPException(
            status_code=503,
            detail="GROQ_API_KEY is not configured",
        )

    return Groq(
        api_key=settings.groq_api_key,
        timeout=45,
        max_retries=1,
    )


# --------------------------------------------------
# Image analysis using Groq vision
# --------------------------------------------------

def inspect_image(data: str | None) -> str:
    if not data:
        return ""

    # Validate and decode the image.
    try:
        if not data.startswith("data:") or "," not in data:
            raise ValueError("Expected a valid image data URL")

        header, payload = data.split(",", 1)
        parts = header.split(";")

        mime = parts[0].removeprefix("data:").lower().strip()

        if mime not in {"image/jpeg", "image/png", "image/webp"}:
            raise ValueError(
                f"Unsupported image type: {mime or 'unknown'}"
            )

        raw = base64.b64decode(payload, validate=True)

        if not raw:
            raise ValueError("The image is empty")

        if len(raw) > 5_000_000:
            raise ValueError("Image must be smaller than 5 MB")

    except (ValueError, IndexError) as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image: {exc}",
        ) from exc

    # Call Groq vision.
    try:
        client = get_groq_client()

        model = settings.groq_vision_model

        if not model:
            raise RuntimeError(
                "GROQ_VISION_MODEL is missing from configuration"
            )

        logger.info("Starting image analysis with model: %s", model)

        result = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Describe objectively visible medical-relevant "
                                "features. Do not diagnose, prescribe treatment, "
                                "or claim that an image rules out disease. "
                                "Mention uncertainty when the image is unclear. "
                                "This is for medical education, not diagnosis."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": data},
                        },
                    ],
                }
            ],
            max_completion_tokens=300,
        )

        description = result.choices[0].message.content or ""

        if not description.strip():
            raise RuntimeError(
                "The vision model returned an empty response"
            )

        return description

    except HTTPException:
        raise

    except Exception as exc:
        # Print the complete error and traceback in your VS Code terminal.
        logger.exception(
            "Groq image analysis failed (model=%s, image_type=%s)",
            settings.groq_vision_model,
            mime,
        )

        raise HTTPException(
            status_code=502,
            detail=(
                "Image analysis failed. Check the server terminal "
                "for the underlying Groq error."
            ),
        ) from exc

# --------------------------------------------------
# Home page
# --------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def home():
    return (
        ROOT / "templates" / "index.html"
    ).read_text(encoding="utf-8")


# --------------------------------------------------
# Health check
# --------------------------------------------------

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "configured": bool(
            settings.groq_api_key
            and settings.pinecone_api_key
        ),
        "tavily": bool(settings.tavily_api_key),
    }


# --------------------------------------------------
# Chat endpoint
# --------------------------------------------------

@app.post("/api/chat")
def chat(
    payload: ChatRequest,
    authorization: str | None = Header(default=None),
    conversation_id: str | None = Header(
        default=None,
        alias="X-Conversation-ID",
    ),
):

    if (
        not settings.groq_api_key
        or not settings.pinecone_api_key
    ):
        raise HTTPException(
            status_code=503,
            detail=(
                "Set GROQ_API_KEY and PINECONE_API_KEY "
                "in your .env file"
            ),
        )

    # Keep only valid conversation messages.
    history = [
        item
        for item in payload.history
        if (
            item.get("role") in ("user", "assistant")
            and isinstance(item.get("content"), str)
        )
    ][-10:]

    thread_id = conversation_id or str(uuid4())

    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    try:
        observation = inspect_image(payload.image_data)

        state = graph.invoke(
            {
                "question": payload.message,
                "history": history,
                "image_description": observation,
                "kb": [],
                "web": [],
                "answer": "",
                "route": "",
            },
            config=config,
        )

        sources = [
            {
                "type": "pdf",
                "page": item.get("page"),
                "score": item.get("score"),
            }
            for item in state.get("kb", [])
        ]

        sources.extend(
            {
                "type": "web",
                "url": item.get("url"),
                "title": item.get("title"),
            }
            for item in state.get("web", [])
        )

        tools_used = ["pinecone"]

        if (
            state.get("route") == "web"
            and settings.tavily_api_key
        ):
            tools_used.append("tavily")

        if observation:
            tools_used.append("vision")

        return {
            "answer": state["answer"],
            "sources": sources,
            "tools": tools_used,
            "conversation_id": thread_id,
        }

    except HTTPException:
        raise

    except Exception:
        # Log details server-side; do not expose secrets to clients.
        import logging

        logging.exception("Chat request failed")

        raise HTTPException(
            status_code=502,
            detail=(
                "AI service failed. Check server logs, "
                "model availability, credentials, Pinecone "
                "configuration, and connectivity."
            ),
        )


# --------------------------------------------------
# Speech-to-text using Groq Whisper
# --------------------------------------------------

@app.post("/api/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    authorization: str | None = Header(default=None),
):

    if not settings.groq_api_key:
        raise HTTPException(
            status_code=503,
            detail="GROQ_API_KEY is not configured",
        )

    allowed_audio_types = {
    "audio/webm",
    "audio/wav",
    "audio/x-wav",
    "audio/mpeg",
    "audio/mp4",
    "audio/ogg",
    "audio/flac",
    "audio/mpga",
    "audio/webm;codecs=opus",
    "audio/ogg;codecs=opus",
    "application/octet-stream",
}

    content_type = (file.content_type or "").lower().strip()

    if content_type and content_type not in allowed_audio_types:
        raise HTTPException(
        status_code=415,
        detail=f"Unsupported audio format: {content_type}",
    )

    data = await file.read(10_000_001)

    if len(data) > 10_000_000:
        raise HTTPException(
            status_code=413,
            detail="Audio exceeds 10 MB",
        )

    if not data:
        raise HTTPException(
            status_code=400,
            detail="Audio file is empty",
        )

    try:
        client = get_groq_client()

        filename = Path(file.filename or "voice.webm").name

        result = client.audio.transcriptions.create(
            model="whisper-large-v3-turbo",
            file=(
                filename,
                data,
                file.content_type or "audio/webm",
            ),
        )

        return {
            "text": result.text,
        }

    except HTTPException:
        raise

    except Exception:
        import logging

        logging.exception("Audio transcription failed")

        raise HTTPException(
            status_code=502,
            detail="Speech transcription unavailable",
        )

    finally:
        await file.close()


# --------------------------------------------------
# Text-to-speech using Groq Orpheus
# --------------------------------------------------

@app.post("/api/speak")
def speak(
    payload: dict,
    authorization: str | None = Header(default=None),
):

    if not settings.groq_api_key:
        raise HTTPException(
            status_code=503,
            detail="GROQ_API_KEY is not configured",
        )

    text = payload.get("text", "")

    if not isinstance(text, str) or not text.strip():
        raise HTTPException(
            status_code=400,
            detail="Missing text",
        )

    # Orpheus currently limits input to 200 characters.
    if len(text) > 200:
        raise HTTPException(
            status_code=400,
            detail=(
                "Text-to-speech supports up to 200 characters "
                "per request. Shorten the text and try again."
            ),
        )

    try:
        client = get_groq_client()

        speech = client.audio.speech.create(
            model="canopylabs/orpheus-v1-english",
            voice="autumn",
            input=text,
            response_format="wav",
        )

        return Response(
            content=speech.read(),
            media_type="audio/wav",
        )

    except HTTPException:
        raise

    except Exception:
        import logging

        logging.exception("Speech synthesis failed")

        raise HTTPException(
            status_code=502,
            detail="Speech synthesis unavailable",
        )
