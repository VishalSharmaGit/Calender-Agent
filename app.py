import asyncio
import io
import wave
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import streamlit as st
from google import genai
from google.genai import types
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build

# ---------------------------------------------------------------- settings
TZ_NAME = "Asia/Kolkata"          # change if you are in another timezone
TZ = ZoneInfo(TZ_NAME)
# Check https://ai.google.dev/gemini-api/docs/live-api/get-started-sdk for the current Live model name
LIVE_MODEL = "gemini-3.1-flash-live-preview"

st.set_page_config(page_title="Voice Assistant", page_icon="🎙️", layout="centered")
st.title("🎙️ Talk to My Calendar")

# Optional password so strangers can't use your calendar (add APP_PASSWORD to secrets)
if "APP_PASSWORD" in st.secrets:
    if st.text_input("Password", type="password") != st.secrets["APP_PASSWORD"]:
        st.stop()

CALENDAR_ID = st.secrets["CALENDAR_ID"]  # your Gmail address (calendar shared with the service account)
client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])


# ---------------------------------------------------------------- calendar
def get_calendar_service():
    creds = Credentials.from_service_account_info(
        dict(st.secrets["gcp_service_account"]),
        scopes=["https://www.googleapis.com/auth/calendar"],
    )
    return build("calendar", "v3", credentials=creds, cache_discovery=False)


def list_events(date: str) -> str:
    try:
        day = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=TZ)
        result = get_calendar_service().events().list(
            calendarId=CALENDAR_ID,
            timeMin=day.isoformat(),
            timeMax=(day + timedelta(days=1)).isoformat(),
            singleEvents=True,
            orderBy="startTime",
        ).execute()
        events = result.get("items", [])
        if not events:
            return f"No events scheduled on {date}."
        return "\n".join(
            f"- {e.get('summary', '(no title)')} at {e['start'].get('dateTime', e['start'].get('date'))}"
            for e in events
        )
    except Exception as e:
        return f"Error reading calendar: {e}"


def add_to_calendar(title: str, date: str, time: str, duration_minutes: int = 60) -> str:
    try:
        start = datetime.strptime(f"{date} {time}", "%Y-%m-%d %H:%M").replace(tzinfo=TZ)
        end = start + timedelta(minutes=int(duration_minutes))
        body = {
            "summary": title,
            "start": {"dateTime": start.isoformat(), "timeZone": TZ_NAME},
            "end": {"dateTime": end.isoformat(), "timeZone": TZ_NAME},
        }
        get_calendar_service().events().insert(calendarId=CALENDAR_ID, body=body).execute()
        return f"Successfully added '{title}' on {date} at {time}."
    except Exception as e:
        return f"Failed to add event: {e}"


FUNCS = {"list_events": list_events, "add_to_calendar": add_to_calendar}

S = types.Schema
T = types.Type
TOOLS = [
    types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="list_events",
                description="Lists calendar events on a date, to check for conflicts.",
                parameters=S(
                    type=T.OBJECT,
                    properties={"date": S(type=T.STRING, description="YYYY-MM-DD")},
                    required=["date"],
                ),
            ),
            types.FunctionDeclaration(
                name="add_to_calendar",
                description="Adds an event. Only call when title, date and time are all known and there is no conflict.",
                parameters=S(
                    type=T.OBJECT,
                    properties={
                        "title": S(type=T.STRING, description="Event title"),
                        "date": S(type=T.STRING, description="YYYY-MM-DD"),
                        "time": S(type=T.STRING, description="HH:MM, 24-hour"),
                        "duration_minutes": S(type=T.INTEGER, description="Default 60"),
                    },
                    required=["title", "date", "time"],
                ),
            ),
        ]
    )
]


# ---------------------------------------------------------------- audio helpers
def wav_to_pcm(wav_bytes: bytes):
    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        return w.readframes(w.getnframes()), w.getframerate()


def pcm_to_wav(pcm: bytes, rate: int = 24000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()


# ---------------------------------------------------------------- Gemini Live
def build_config() -> types.LiveConnectConfig:
    now = datetime.now(TZ)
    history = "\n".join(st.session_state.history[-20:]) or "(new conversation)"
    instruction = f"""
You are my strict personal calendar assistant. Current date/time: {now:%A, %B %d, %Y %H:%M} ({TZ_NAME}).
Your job is to book events for me. You need a Title, Date and Time.
If any detail is missing, do NOT call add_to_calendar. Ask me one direct question to get it.
Before booking, call list_events for that date. If there is a conflict, tell me and ask what to do.
Keep spoken replies short.

Conversation so far (each recording is a new session, so use this as memory):
{history}
"""
    return types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        system_instruction=instruction,
        tools=TOOLS,
        input_audio_transcription=types.AudioTranscriptionConfig(),
        output_audio_transcription=types.AudioTranscriptionConfig(),
    )


async def run_agent(pcm: bytes, rate: int):
    reply_pcm = bytearray()
    heard, said = [], []
    tool_called = False
    silence = b"\x00" * (rate * 2)  # 1 second of silence so the end of speech is detected

    async with client.aio.live.connect(model=LIVE_MODEL, config=build_config()) as session:
        await session.send_realtime_input(
            audio=types.Blob(data=pcm + silence, mime_type=f"audio/pcm;rate={rate}")
        )
        await session.send_realtime_input(audio_stream_end=True)

        async for msg in session.receive():
            if msg.tool_call:
                tool_called = True
                responses = []
                for call in msg.tool_call.function_calls:
                    fn = FUNCS.get(call.name)
                    result = fn(**(call.args or {})) if fn else "Unknown tool"
                    responses.append(
                        types.FunctionResponse(id=call.id, name=call.name, response={"result": result})
                    )
                await session.send_tool_response(function_responses=responses)

            sc = msg.server_content
            if sc:
                if sc.input_transcription and sc.input_transcription.text:
                    heard.append(sc.input_transcription.text)
                if sc.output_transcription and sc.output_transcription.text:
                    said.append(sc.output_transcription.text)
                if sc.model_turn:
                    for part in sc.model_turn.parts:
                        if part.inline_data and part.inline_data.data:
                            reply_pcm.extend(part.inline_data.data)
                if sc.turn_complete:
                    if tool_called and not reply_pcm:
                        continue  # model still has to answer after the tool result
                    break

    return bytes(reply_pcm), "".join(heard).strip(), "".join(said).strip()


# ---------------------------------------------------------------- UI
if "history" not in st.session_state:
    st.session_state.history = []
    st.session_state.n = 0
    st.session_state.reply = None

st.write("Record a message, then stop. The assistant replies out loud and may ask you questions.")

audio = st.audio_input("Record a voice message", sample_rate=16000, key=f"mic_{st.session_state.n}")

if audio is not None:
    with st.spinner("Agent is thinking..."):
        try:
            pcm, rate = wav_to_pcm(audio.read())
            reply_pcm, heard, said = asyncio.run(asyncio.wait_for(run_agent(pcm, rate), timeout=60))
            if heard:
                st.session_state.history.append(f"Me: {heard}")
            if said:
                st.session_state.history.append(f"Assistant: {said}")
            st.session_state.reply = pcm_to_wav(reply_pcm) if reply_pcm else None
            st.session_state.error = None
        except Exception as e:
            st.session_state.reply = None
            st.session_state.error = f"Error: {e}"
    st.session_state.n += 1  # resets the mic widget for the next turn
    st.rerun()

if st.session_state.get("error"):
    st.error(st.session_state.error)

if st.session_state.reply:
    st.audio(st.session_state.reply, format="audio/wav", autoplay=True)

for line in st.session_state.history:
    st.write(line)

if st.button("New conversation"):
    st.session_state.history = []
    st.session_state.reply = None
    st.rerun()
