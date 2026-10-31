import asyncio
import base64
import html
import io
import time
import wave
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import streamlit as st
import streamlit.components.v1 as components
from google import genai
from google.genai import types
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build

# ---------------------------------------------------------------- settings
TZ_NAME = "Asia/Kolkata"          # change if you are in another timezone
TZ = ZoneInfo(TZ_NAME)
# Check https://ai.google.dev/gemini-api/docs/live-api/get-started-sdk for the current Live model name
LIVE_MODEL = "gemini-3.1-flash-live-preview"

st.set_page_config(page_title="Voice Calendar", page_icon="🎙️", layout="centered")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@500;700&display=swap');
.stApp{background:#0a0716;font-family:'Plus Jakarta Sans',system-ui,sans-serif;color:#fff}
.stApp::before,.stApp::after{content:"";position:fixed;border-radius:50%;filter:blur(80px);z-index:0;pointer-events:none}
.stApp::before{top:60px;left:-100px;width:320px;height:320px;background:#19d3c5;opacity:.4}
.stApp::after{bottom:80px;right:-100px;width:340px;height:340px;background:#8b3dff;opacity:.55}
header[data-testid="stHeader"]{background:transparent}
#MainMenu,footer{visibility:hidden}
.block-container{max-width:430px;padding:1.2rem 1rem 3rem;position:relative;z-index:1}
.g{background:linear-gradient(135deg,rgba(255,255,255,.14),rgba(255,255,255,.04));backdrop-filter:blur(22px) saturate(160%);-webkit-backdrop-filter:blur(22px) saturate(160%);border:1px solid rgba(255,255,255,.18);border-radius:24px;padding:18px 18px 18px 22px;margin-bottom:14px;box-shadow:0 10px 40px rgba(0,0,0,.35),inset 0 1px 0 rgba(255,255,255,.22)}
.h1{font-size:30px;font-weight:700;line-height:1.1}
.mut{font-size:14px;color:rgba(255,255,255,.75)}
.hd{margin-bottom:14px}
.line{border-left:2px solid rgba(255,255,255,.2);padding-left:18px;display:flex;flex-direction:column;gap:16px}
.t{font-size:12px;color:rgba(255,255,255,.75)}.t.nx{color:#5eead4}
.n{font-size:16px;font-weight:700}
.who{font-size:12px;color:rgba(255,255,255,.75);margin-top:10px}.who.a{color:#5eead4}
.stButton>button{background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.2);color:#fff;border-radius:14px}
[data-testid="stExpander"]{background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.14);border-radius:18px}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

if "APP_PASSWORD" in st.secrets:
    if st.text_input("Password", type="password") != st.secrets["APP_PASSWORD"]:
        st.stop()

CALENDAR_ID = st.secrets["CALENDAR_ID"]
client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
orb = components.declare_component("voice_orb", path=str(Path(__file__).parent / "orb"))


# ---------------------------------------------------------------- calendar
def get_calendar_service():
    creds = Credentials.from_service_account_info(
        dict(st.secrets["gcp_service_account"]),
        scopes=["https://www.googleapis.com/auth/calendar"],
    )
    return build("calendar", "v3", credentials=creds, cache_discovery=False)


def _raw_events(date: str):
    day = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=TZ)
    result = get_calendar_service().events().list(
        calendarId=CALENDAR_ID,
        timeMin=day.isoformat(),
        timeMax=(day + timedelta(days=1)).isoformat(),
        singleEvents=True,
        orderBy="startTime",
    ).execute()
    return result.get("items", [])


@st.cache_data(ttl=30, show_spinner=False)
def fetch_events(date: str):
    try:
        out = []
        for e in _raw_events(date):
            s = e["start"]
            if "dateTime" in s:
                dt = datetime.fromisoformat(s["dateTime"]).astimezone(TZ)
                out.append({"label": f"{dt:%I:%M %p}".lstrip("0"), "title": e.get("summary", "(no title)"), "dt": dt})
            else:
                out.append({"label": "All day", "title": e.get("summary", "(no title)"), "dt": None})
        return out
    except Exception:
        return None


def list_events(date: str) -> str:
    try:
        events = _raw_events(date)
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
        created = get_calendar_service().events().insert(calendarId=CALENDAR_ID, body=body).execute()
        fetch_events.clear()
        return (f"Successfully added '{title}' on {date} at {time}. "
                f"Event id: {created.get('id')} Link: {created.get('htmlLink')}")
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
                parameters=S(type=T.OBJECT, properties={"date": S(type=T.STRING, description="YYYY-MM-DD")}, required=["date"]),
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
    heard, said, log = [], [], []
    tool_called = False
    silence = b"\x00" * (rate * 2)

    async with client.aio.live.connect(model=LIVE_MODEL, config=build_config()) as session:
        await session.send_realtime_input(audio=types.Blob(data=pcm + silence, mime_type=f"audio/pcm;rate={rate}"))
        await session.send_realtime_input(audio_stream_end=True)

        async for msg in session.receive():
            if msg.tool_call:
                tool_called = True
                responses = []
                for call in msg.tool_call.function_calls:
                    fn = FUNCS.get(call.name)
                    result = fn(**(call.args or {})) if fn else "Unknown tool"
                    log.append(f"{call.name}({dict(call.args or {})}) -> {result}")
                    responses.append(types.FunctionResponse(id=call.id, name=call.name, response={"result": result}))
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
                        continue
                    break

    if not tool_called:
        log.append("No calendar tool was called this turn.")
    return bytes(reply_pcm), "".join(heard).strip(), "".join(said).strip(), log


def handle_audio(pcm: bytes, rate: int):
    """Send a recording to Gemini and store the results in session state."""
    try:
        reply_pcm, heard, said, log = asyncio.run(asyncio.wait_for(run_agent(pcm, rate), timeout=60))
        st.session_state.tool_log.extend(log)
        if heard:
            st.session_state.history.append(f"Me: {heard}")
        if said:
            st.session_state.history.append(f"Assistant: {said}")
        if reply_pcm:
            st.session_state.reply_b64 = base64.b64encode(pcm_to_wav(reply_pcm)).decode()
            st.session_state.reply_id = str(time.time_ns())
        else:
            st.session_state.reply_b64 = None
            st.session_state.reply_id = None
        st.session_state.error = None
    except Exception as e:
        st.session_state.reply_b64 = None
        st.session_state.reply_id = None
        st.session_state.error = f"Error: {e}"
    st.session_state.n += 1


# ---------------------------------------------------------------- state
for k, v in {"history": [], "tool_log": [], "n": 0, "last_id": None, "reply_b64": None, "reply_id": None, "error": None}.items():
    st.session_state.setdefault(k, v)

# ---------------------------------------------------------------- UI: header + today
now = datetime.now(TZ)
events = fetch_events(now.strftime("%Y-%m-%d"))
count = "calendar unavailable" if events is None else f"{len(events)} event{'s' if len(events) != 1 else ''}"
st.markdown(f'<div class="hd"><div class="h1">Today</div><div class="mut">{now:%a, %d %b} · {count}</div></div>', unsafe_allow_html=True)

if events is None:
    body = '<div class="mut">Couldn’t load your calendar. Open Details and press Test calendar connection.</div>'
elif not events:
    body = '<div class="mut">Nothing scheduled today. Tap the orb and say “Book dinner tomorrow at 7:30”.</div>'
else:
    nxt = next((i for i, e in enumerate(events) if e["dt"] and e["dt"] >= now), None)
    rows = []
    for i, e in enumerate(events):
        flag = " · up next" if i == nxt else ""
        cls = "t nx" if i == nxt else "t"
        rows.append(f'<div><div class="{cls}">{e["label"]}{flag}</div><div class="n">{html.escape(e["title"])}</div></div>')
    body = '<div class="line">' + "".join(rows) + "</div>"
st.markdown(f'<div class="g">{body}</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------- UI: conversation
if st.session_state.error:
    st.markdown(f'<div class="g"><div class="who a">Something went wrong</div>{html.escape(st.session_state.error)}</div>', unsafe_allow_html=True)
elif st.session_state.history:
    parts = []
    for line in st.session_state.history[-4:]:
        who, _, text = line.partition(": ")
        label = "You" if who == "Me" else "Assistant"
        parts.append(f'<div class="who{" a" if label == "Assistant" else ""}">{label}</div>{html.escape(text)}')
    st.markdown('<div class="g">' + "".join(parts) + "</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------- UI: the orb
val = orb(token=st.session_state.n, reply_id=st.session_state.reply_id, reply_b64=st.session_state.reply_b64, key="orb", default=None)
if val and val.get("id") != st.session_state.last_id:
    st.session_state.last_id = val["id"]
    handle_audio(base64.b64decode(val["pcm_b64"]), int(val["rate"]))
    st.rerun()

# ---------------------------------------------------------------- UI: details
with st.expander("Details"):
    st.caption("Calendar activity (real results)")
    if st.session_state.tool_log:
        for item in st.session_state.tool_log[-10:]:
            st.code(item, language=None)
    else:
        st.caption("Nothing yet.")
    if st.button("Test calendar connection"):
        st.write(f"Calendar ID in use: {CALENDAR_ID}")
        st.code(list_events(now.strftime("%Y-%m-%d")), language=None)
    st.caption("Orb not recording? Use the basic recorder instead:")
    fb = st.audio_input("Basic recorder", sample_rate=16000, key=f"fb_{st.session_state.n}")
    if fb is not None:
        pcm, rate = wav_to_pcm(fb.read())
        handle_audio(pcm, rate)
        st.rerun()
    if st.session_state.reply_b64:
        st.audio(base64.b64decode(st.session_state.reply_b64), format="audio/wav")
    if st.button("New conversation"):
        st.session_state.history = []
        st.session_state.reply_b64 = None
        st.session_state.reply_id = None
        st.session_state.error = None
        st.rerun()