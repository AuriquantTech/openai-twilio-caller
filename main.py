import os
import json
import base64
import asyncio
import logging
import re
import secrets
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
import websockets
from fastapi import FastAPI, WebSocket, Request, HTTPException, Header, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.websockets import WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from pydantic import BaseModel, Field, validator
from twilio.twiml.voice_response import VoiceResponse, Connect, Say, Stream
from dotenv import load_dotenv
from twilio.rest import Client
from fastapi.responses import FileResponse



# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

# Initialize Twilio client
try:
    twilio_client = Client(
        os.getenv("TWILIO_ACCOUNT_SID"),
        os.getenv("TWILIO_AUTH_TOKEN")
    )
except Exception as e:
    logger.error(f"Failed to initialize Twilio client: {e}")
    raise

TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")

# Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
API_KEY = os.getenv('API_KEY')
PORT = int(os.getenv('PORT', 5050))
TEMPERATURE = float(os.getenv('TEMPERATURE', 0.8))

# Constants
HTTP_STATUS_OK = 200
HTTP_STATUS_FORBIDDEN = 403
HTTP_STATUS_UNAUTHORIZED = 401
HTTP_STATUS_BAD_REQUEST = 400
PHONE_REGEX = r'^\+?[1-9]\d{1,14}$'

SYSTEM_MESSAGE = (
    "You are a friendly and professional business development representative calling on behalf of Midrat Business Solutions. "
    "Your goal is to have a natural, warm conversation while collecting important contact information. "
    
    "CONVERSATION FLOW:\n"
    "1. Start with a warm greeting and briefly introduce yourself and Midrat Business Solutions\n"
    "2. Ask if this is a good time to talk (if not, offer to call back)\n"
    "3. Naturally collect the following information during the conversation:\n"
    "   - Their full name\n"
    "   - Their job title or role\n"
    "   - Their city/location\n"
    "   - Their email address (for sending follow-up information)\n"
    
    "IMPORTANT RULES:\n"
    "- Sound like a real person, not a bot or scripted agent\n"
    "- Keep your tone conversational, warm, and professional\n"
    "- Ask for one piece of information at a time, naturally woven into conversation\n"
    "- If they seem hesitant, reassure them this is just for follow-up information\n"
    "- Confirm each detail once, briefly (e.g., 'Great, so that's john@example.com, correct?')\n"
    "- Do NOT rush - build rapport before asking for information\n"
    "- If they ask what this is about, explain we provide business solutions and want to send them relevant information\n"
    "- When you have all 4 pieces of information, thank them warmly and let them know they'll receive more details via email\n"
    "- Keep sentences short and clear\n"
    "- Never guess or make up information - if unclear, politely ask again\n"
    
    "ENDING THE CALL:\n"
    "- If the caller says goodbye, bye, thanks, or indicates they want to end (like 'cut the call', 'hang up', 'that's all'), politely wrap up\n"
    "- Always end warmly: thank them for their time\n"
    "- Use the end_call function when the conversation is clearly over\n"
    "- Don't abruptly end - give a proper closing statement first\n"
    
    "ONCE YOU HAVE ALL INFORMATION (name, job title, city, email), use the save_lead function to store it."
)

# Validate temperature range for OpenAI Realtime API
if TEMPERATURE < 0.6 or TEMPERATURE > 1.2:
    logger.warning(f"Temperature {TEMPERATURE} is outside recommended range (0.6-1.2). Adjusting to 0.8")
    TEMPERATURE = 0.8




VOICE = 'verse'
LOG_EVENT_TYPES = [
    'error', 'response.content.done', 'rate_limits.updated',
    'response.done', 'input_audio_buffer.committed',
    'input_audio_buffer.speech_stopped', 'input_audio_buffer.speech_started',
    'session.created', 'session.updated'
]
SHOW_TIMING_MATH = False

# Pydantic models for request validation
class CallRequest(BaseModel):
    to: str = Field(..., description="Phone number to call")
    
    @validator('to')
    def validate_phone(cls, v: str) -> str:
        if not re.match(PHONE_REGEX, v):
            raise ValueError('Invalid phone number format. Use E.164 format (e.g., +1234567890)')
        return v

class CallResponse(BaseModel):
    status: str
    to: str
    call_sid: str

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None

class LeadData(BaseModel):
    name: str
    job_title: str
    city: str
    email: str
    phone_number: Optional[str] = None
    call_sid: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    
    @validator('email')
    def validate_email(cls, v: str) -> str:
        email_regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_regex, v):
            raise ValueError('Invalid email format')
        return v.lower()

# Security dependency
async def verify_api_key(x_api_key: Optional[str] = Header(None)) -> str:
    """Verify API key for protected endpoints."""
    if not API_KEY:
        logger.error("API_KEY not configured in environment")
        raise HTTPException(
            status_code=HTTP_STATUS_UNAUTHORIZED,
            detail="API authentication not configured"
        )
    
    if not x_api_key or x_api_key != API_KEY:
        logger.warning("Invalid API key attempt")
        raise HTTPException(
            status_code=HTTP_STATUS_UNAUTHORIZED,
            detail="Invalid API key"
        )
    return x_api_key

# Initialize FastAPI app
app = FastAPI(
    title="Twilio + OpenAI Realtime Voice Agent",
    description="Real-time AI voice agent using Twilio and OpenAI Realtime API",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure with your domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

if not OPENAI_API_KEY:
    raise ValueError('Missing the OpenAI API key. Please set it in the .env file.')

@app.get("/health", response_class=JSONResponse)
async def health_check() -> Dict[str, str]:
    """Health check endpoint for monitoring."""
    return {
        "status": "healthy",
        "service": "twilio-openai-voice-agent"
    }

app = FastAPI()

if not OPENAI_API_KEY:
    raise ValueError('Missing the OpenAI API key. Please set it in the .env file.')

# Lead storage setup
LEADS_FILE = Path("leads.json")

def save_lead_to_file(lead_data: Dict[str, Any]) -> bool:
    """Save lead data to JSON file."""
    try:
        # Load existing leads
        leads = []
        if LEADS_FILE.exists():
            with open(LEADS_FILE, 'r') as f:
                leads = json.load(f)
        
        # Add new lead
        leads.append(lead_data)
        
        # Save back to file
        with open(LEADS_FILE, 'w') as f:
            json.dump(leads, f, indent=2)
        
        logger.info(f"Lead saved: {lead_data.get('name')} - {lead_data.get('email')}")
        return True
    except Exception as e:
        logger.error(f"Failed to save lead: {e}")
        return False

def get_base_url(request: Request) -> str:
    """Get the base URL for Twilio callbacks."""
    # Check if we're behind a proxy
    forwarded_proto = request.headers.get("x-forwarded-proto", "https")
    host = request.url.hostname
    
    return f"{forwarded_proto}://{host}"

@app.get("/", response_class=JSONResponse)
async def index_page() -> Dict[str, str]:
    """Root endpoint."""
    return {
        "message": "Twilio Media Stream Server is running!",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.api_route("/incoming-call", methods=["GET", "POST"])
async def handle_incoming_call(request: Request) -> HTMLResponse:
    """Handle incoming call and return TwiML response to connect to Media Stream."""
    try:
        response = VoiceResponse()
        # <Say> punctuation to improve text-to-speech flow
        response.say(
            "Please wait while we connect your call to an agent from Midrat Business Solutions",
            voice="Google.en-IN-Neural2-D"
        )
        response.pause(length=1)
        response.say(   
            "Call Connected!",
            voice="Google.en-US-Chirp3-HD-Aoede"
        )
        
        # Get base URL and construct WebSocket URL
        base_url = get_base_url(request)
        ws_protocol = "wss" if base_url.startswith("https") else "ws"
        ws_url = base_url.replace("https://", "").replace("http://", "")
        
        connect = Connect()
        connect.stream(url=f'{ws_protocol}://{ws_url}/media-stream')
        response.append(connect)
        
        logger.info(f"Incoming call handled, connecting to media stream at {ws_url}")
        return HTMLResponse(content=str(response), media_type="application/xml")
    except Exception as e:
        logger.error(f"Error handling incoming call: {e}")
        raise HTTPException(status_code=500, detail="Failed to handle incoming call")

@app.post("/call", response_model=CallResponse)
@limiter.limit("5/minute")
async def make_outbound_call(
    request: Request,
    call_request: CallRequest,
    api_key: str = Depends(verify_api_key)
) -> CallResponse:
    """
    Initiates an outbound call to a phone number
    and connects it to the AI agent.
    """
    try:
        base_url = get_base_url(request)
        callback_url = f"{base_url}/incoming-call"
        
        logger.info(f"Initiating outbound call to {call_request.to}")
        logger.info(f"Using callback URL: {callback_url}")
        
        call = twilio_client.calls.create(
            to=call_request.to,
            from_=TWILIO_PHONE_NUMBER,
            url=callback_url
        )
        
        logger.info(f"Call created successfully: {call.sid}")
        
        return CallResponse(
            status="calling",
            to=call_request.to,
            call_sid=call.sid
        )
    except Exception as e:
        logger.error(f"Error making outbound call: {e}")
        raise HTTPException(
            status_code=HTTP_STATUS_BAD_REQUEST,
            detail=f"Failed to initiate call: {str(e)}"
        )

@app.get("/browser")
async def browser_call(api_key: str = Depends(verify_api_key)):
    """Serve the browser-based calling interface."""
    return FileResponse("call.html")


@app.websocket("/media-stream")
async def handle_media_stream(websocket: WebSocket) -> None:
    """Handle WebSocket connections between Twilio and OpenAI."""
    logger.info("Client connected to media stream")
    await websocket.accept()

    openai_ws = None
    try:
        async with websockets.connect(
            f"wss://api.openai.com/v1/realtime?model=gpt-realtime-mini&temperature={TEMPERATURE}",
            additional_headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}"
            },
            ping_interval=20,
            ping_timeout=10
        ) as openai_ws:
            await initialize_session(openai_ws)

            # Connection specific state
            stream_sid: Optional[str] = None
            call_sid: Optional[str] = None
            latest_media_timestamp: int = 0
            last_assistant_item: Optional[str] = None
            mark_queue: List[str] = []
            response_start_timestamp_twilio: Optional[int] = None
            
            async def receive_from_twilio() -> None:
                """Receive audio data from Twilio and send it to the OpenAI Realtime API."""
                nonlocal stream_sid, call_sid, latest_media_timestamp
                try:
                    async for message in websocket.iter_text():
                        data = json.loads(message)
                        
                        if data['event'] == 'media':
                            if openai_ws and not openai_ws.closed:
                                latest_media_timestamp = int(data['media']['timestamp'])
                                audio_append = {
                                    "type": "input_audio_buffer.append",
                                    "audio": data['media']['payload']
                                }
                                await openai_ws.send(json.dumps(audio_append))
                        
                        elif data['event'] == 'start':
                            stream_sid = data['start']['streamSid']
                            call_sid = data['start'].get('callSid', stream_sid)
                            logger.info(f"Incoming stream has started. Stream SID: {stream_sid}, Call SID: {call_sid}")
                            response_start_timestamp_twilio = None
                            latest_media_timestamp = 0
                            last_assistant_item = None
                        
                        elif data['event'] == 'mark':
                            if mark_queue:
                                mark_queue.pop(0)
                                
                except WebSocketDisconnect:
                    logger.info("Client disconnected from Twilio")
                except Exception as e:
                    logger.error(f"Error in receive_from_twilio: {e}")
                finally:
                    if openai_ws and not openai_ws.closed:
                        await openai_ws.close()

            async def send_to_twilio() -> None:
                """Receive events from the OpenAI Realtime API, send audio back to Twilio."""
                nonlocal stream_sid, last_assistant_item, response_start_timestamp_twilio
                try:
                    async for openai_message in openai_ws:
                        response = json.loads(openai_message)
                        
                        if response['type'] in LOG_EVENT_TYPES:
                            logger.info(f"Received event: {response['type']}")
                            logger.debug(f"Event details: {response}")

                        # Handle function calls from OpenAI to save lead data
                        if response.get('type') == 'response.function_call_arguments.done':
                            function_name = response.get('name')
                            call_id = response.get('call_id')
                            
                            if function_name == 'save_lead':
                                try:
                                    # Parse the function arguments
                                    arguments = json.loads(response.get('arguments', '{}'))
                                    
                                    # Add call metadata
                                    lead_data = {
                                        **arguments,
                                        'call_sid': call_sid or stream_sid,
                                        'timestamp': datetime.now().isoformat()
                                    }
                                    
                                    # Save the lead
                                    save_lead_to_file(lead_data)
                                    logger.info(f"✅ Lead collected and saved: {arguments.get('name')}")
                                    
                                    # Send function result back to OpenAI
                                    function_result = {
                                        "type": "conversation.item.create",
                                        "item": {
                                            "type": "function_call_output",
                                            "call_id": call_id,
                                            "output": json.dumps({"status": "success", "message": "Lead information saved successfully"})
                                        }
                                    }
                                    await openai_ws.send(json.dumps(function_result))
                                except Exception as e:
                                    logger.error(f"Error saving lead from function call: {e}")
                            
                            elif function_name == 'end_call':
                                try:
                                    # Parse the function arguments
                                    arguments = json.loads(response.get('arguments', '{}'))
                                    reason = arguments.get('reason', 'AI requested end')
                                    
                                    logger.info(f"📞 Ending call - Reason: {reason}")
                                    
                                    # Send function result back to OpenAI
                                    function_result = {
                                        "type": "conversation.item.create",
                                        "item": {
                                            "type": "function_call_output",
                                            "call_id": call_id,
                                            "output": json.dumps({"status": "success", "message": "Call will end"})
                                        }
                                    }
                                    await openai_ws.send(json.dumps(function_result))
                                    
                                    # Wait a moment for final audio to be sent
                                    await asyncio.sleep(2)
                                    
                                    # Hang up the call using Twilio
                                    if call_sid:
                                        try:
                                            # Update call status to completed
                                            twilio_client.calls(call_sid).update(status='completed')
                                            logger.info(f"✅ Call {call_sid} ended successfully")
                                        except Exception as twilio_error:
                                            logger.error(f"Error ending Twilio call: {twilio_error}")
                                    
                                    # Close WebSocket connections
                                    if openai_ws and not openai_ws.closed:
                                        await openai_ws.close()
                                    if websocket:
                                        await websocket.close()
                                        
                                except Exception as e:
                                    logger.error(f"Error ending call: {e}")

                        if response.get('type') == 'response.output_audio.delta' and 'delta' in response:
                            audio_payload = base64.b64encode(base64.b64decode(response['delta'])).decode('utf-8')
                            audio_delta = {
                                "event": "media",
                                "streamSid": stream_sid,
                                "media": {
                                    "payload": audio_payload
                                }
                            }
                            await websocket.send_json(audio_delta)

                            if response.get("item_id") and response["item_id"] != last_assistant_item:
                                response_start_timestamp_twilio = latest_media_timestamp
                                last_assistant_item = response["item_id"]
                                if SHOW_TIMING_MATH:
                                    logger.debug(f"Setting start timestamp for new response: {response_start_timestamp_twilio}ms")

                            await send_mark(websocket, stream_sid)

                        # Trigger an interruption. Your use case might work better using `input_audio_buffer.speech_stopped`, or combining the two.
                        if response.get('type') == 'input_audio_buffer.speech_started':
                            logger.info("Speech started detected.")
                            if last_assistant_item:
                                logger.info(f"Interrupting response with id: {last_assistant_item}")
                                await handle_speech_started_event()
                                
                except Exception as e:
                    logger.error(f"Error in send_to_twilio: {e}")

            async def handle_speech_started_event() -> None:
                """Handle interruption when the caller's speech starts."""
                nonlocal response_start_timestamp_twilio, last_assistant_item
                logger.info("Handling speech started event.")
                
                if mark_queue and response_start_timestamp_twilio is not None:
                    elapsed_time = latest_media_timestamp - response_start_timestamp_twilio
                    if SHOW_TIMING_MATH:
                        logger.debug(f"Calculating elapsed time for truncation: {latest_media_timestamp} - {response_start_timestamp_twilio} = {elapsed_time}ms")

                    if last_assistant_item:
                        if SHOW_TIMING_MATH:
                            logger.debug(f"Truncating item with ID: {last_assistant_item}, Truncated at: {elapsed_time}ms")

                        truncate_event = {
                            "type": "conversation.item.truncate",
                            "item_id": last_assistant_item,
                            "content_index": 0,
                            "audio_end_ms": elapsed_time
                        }
                        await openai_ws.send(json.dumps(truncate_event))

                    await websocket.send_json({
                        "event": "clear",
                        "streamSid": stream_sid
                    })

                    mark_queue.clear()
                    last_assistant_item = None
                    response_start_timestamp_twilio = None

            async def send_mark(connection: WebSocket, sid: Optional[str]) -> None:
                """Send mark event to track audio playback."""
                if sid:
                    mark_event = {
                        "event": "mark",
                        "streamSid": sid,
                        "mark": {"name": "responsePart"}
                    }
                    await connection.send_json(mark_event)
                    mark_queue.append('responsePart')

            await asyncio.gather(receive_from_twilio(), send_to_twilio())
            
    except Exception as e:
        logger.error(f"Error in media stream handler: {e}")
    finally:
        logger.info("Media stream connection closed")

async def send_initial_conversation_item(openai_ws) -> None:
    """Send initial conversation item if AI talks first."""
    initial_conversation_item = {
        "type": "conversation.item.create",
        "item": {
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "Greet the user professionally and ask how you can help them today."
                }
            ]
        }
    }
    await openai_ws.send(json.dumps(initial_conversation_item))
    await openai_ws.send(json.dumps({"type": "response.create"}))


async def initialize_session(openai_ws) -> None:
    """Control initial session with OpenAI."""
    session_update = {
        "type": "session.update",
        "session": {
            "type": "realtime",
            "model": "gpt-realtime-mini",
            "output_modalities": ["audio", "text"],
            "audio": {
                "input": {
                    "format": {"type": "audio/pcmu"},
                    "turn_detection": {"type": "server_vad"}
                },
                "output": {
                    "format": {"type": "audio/pcmu"},
                    "voice": VOICE
                }
            },
            "instructions": SYSTEM_MESSAGE,
            "tools": [
                {
                    "type": "function",
                    "name": "save_lead",
                    "description": "Save the lead information when you have collected all required details (name, job_title, city, email). Call this function once you have confirmed all four pieces of information with the caller.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "The full name of the lead"
                            },
                            "job_title": {
                                "type": "string",
                                "description": "The job title or role of the lead"
                            },
                            "city": {
                                "type": "string",
                                "description": "The city or location of the lead"
                            },
                            "email": {
                                "type": "string",
                                "description": "The email address of the lead"
                            }
                        },
                        "required": ["name", "job_title", "city", "email"]
                    }
                },
                {
                    "type": "function",
                    "name": "end_call",
                    "description": "End the phone call when the conversation is complete. Use this when: the caller says goodbye/bye/thanks, indicates they want to hang up, or after you've thanked them for their information. Always give a warm closing statement BEFORE calling this function.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "reason": {
                                "type": "string",
                                "description": "Brief reason for ending (e.g., 'caller said goodbye', 'information collected', 'caller requested to end')"
                            }
                        },
                        "required": ["reason"]
                    }
                }
            ],
            "tool_choice": "auto"
        }
    }
    logger.info('Sending session update to OpenAI')
    logger.debug(f'Session config: {json.dumps(session_update)}')
    await openai_ws.send(json.dumps(session_update))

    # Uncomment the next line to have the AI speak first
    # await send_initial_conversation_item(openai_ws)

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting server on port {PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
