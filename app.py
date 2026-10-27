import os
import asyncio
import streamlit as st
from google import genai
from google.genai import types
from googleapiclient.discovery import build
from google.oauth2.service_account import Credentials

# 1. System Prompt & Instructions
SYSTEM_INSTRUCTION = """
You are my strict personal calendar assistant. Today's date is Tuesday, October 27, 2026. 
Your job is to book events for me. When I mention an event, check if you have the Title, Date, and Time. 
If any detail is missing, or if you notice an obvious conflict by checking the existing schedule, 
DO NOT call the add_to_calendar tool. Instead, pause and ask me a direct cross-question to get the missing info.
"""

# 2. Define Live Real-Time Calendar Integrations
def get_calendar_service():
    # Authenticates using your deployment environment secret keys
    scopes = ['https://googleapis.com']
    creds = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=scopes)
    return build('calendar', 'v3', credentials=creds)

def list_events(date: str) -> str:
    """Lists calendar events on a given date to check for schedule conflicts."""
    try:
        service = get_calendar_service()
        time_min = f"{date}T00:00:00Z"
        time_max = f"{date}T23:59:59Z"
        events_result = service.events().list(
            calendarId='primary', timeMin=time_min, timeMax=time_max,
            singleEvents=True, orderBy='startTime'
        ).execute()
        events = events_result.get('items', [])
        if not events:
            return f"No events scheduled on {date}."
        
        summary = [f"- {e['summary']} at {e['start'].get('dateTime', e['start'].get('date'))}" for e in events]
        return "\n".join(summary)
    except Exception as e:
        return f"Error reading calendar: {str(e)}"

def add_to_calendar(title: str, date: str, time: str) -> str:
    """Creates a new event item in the calendar once all details are fully cleared."""
    try:
        service = get_calendar_service()
        event_time = f"{date}T{time}:00"
        event = {
            'summary': title,
            'start': {'dateTime': event_time, 'timeZone': 'UTC'},
            'end': {'dateTime': f"{date}T{int(time.split(':')[0])+1}:{time.split(':')[1]}:00", 'timeZone': 'UTC'}
        }
        created_event = service.events().insert(calendarId='primary', body=event).execute()
        return f"Successfully added '{title}' on {date} at {time}!"
    except Exception as e:
        return f"Failed to add event: {str(e)}"

# 3. Streamlit Interface & Multimodal Audio Session Loop
st.set_page_config(page_title="Voice Assistant", page_icon="🎙️", layout="centered")
st.title("🎙️ Talk to My Calendar")
st.write("Click mic button and speak directly to update your calendar.")

# Initialize the Gemini Live Client
client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

# Setup tool definitions matching your AI Studio parameters
tools = [list_events, add_to_calendar]

# FIX: Pass the modality directly as a list containing the string "AUDIO"
config = types.LiveConnectConfig(
    response_modalities=["AUDIO"],
    generation_config=types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        tools=tools
    )
)



# Streamlit Audio input connection block
audio_value = st.audio_input("Record audio command")

async def run_live_voice_agent(audio_bytes):
    # Initializes a real-time live bi-directional audio stream session
    async with client.aio.live.connect(model="gemini-2.0-flash-exp", config=config) as session:
        # Send user phone audio stream block
        await session.send(input=audio_bytes, end_of_turn=True)
        
        # Monitor the stream back from Gemini
        async for response in session.receive():
            server_content = response.server_content
            if server_content is not None:
                model_turn = server_content.model_turn
                if model_turn is not None:
                    for part in model_turn.parts:
                        # If Gemini decides to speak back, play the audio data chunk
                        if part.inline_data:
                            st.audio(part.inline_data.data, format="audio/wav")
            
            # If Gemini determines it is ready to run the tool, automatically execute it
            if response.tool_call is not None:
                for call in response.tool_call.function_calls:
                    if call.name == "list_events":
                        res = list_events(**call.args)
                    elif call.name == "add_to_calendar":
                        res = add_to_calendar(**call.args)
                    
                    # Return results back into the loop conversation track
                    await session.send(input=types.LiveClientToolResponse(
                        function_responses=[types.FunctionResponse(name=call.name, id=call.id, response={"result": res})]
                    ))

if audio_value:
    st.info("Agent is processing and responding out loud...")
    asyncio.run(run_live_voice_agent(audio_value.read()))
