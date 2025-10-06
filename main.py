import asyncio
import base64
import json
import logging
import lmdb
import os
import re
import hashlib
import tempfile
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Union

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from gemini_webapi import GeminiClient, set_log_level
from gemini_webapi.constants import Model
from gemini_webapi.types import Gem
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
set_log_level("INFO")

app = FastAPI(title="Gemini API FastAPI Server")

# Add CORS middleware
app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)

# Global client
gemini_client = None

# Authentication credentials
SECURE_1PSID = os.environ.get("SECURE_1PSID", "")
SECURE_1PSIDTS = os.environ.get("SECURE_1PSIDTS", "")
API_KEY = os.environ.get("API_KEY", "")

# LMDB for gem caching
GEMS_DB_PATH = "gemini_db"
if not os.path.exists(GEMS_DB_PATH):
    os.makedirs(GEMS_DB_PATH)
gems_env = lmdb.open(GEMS_DB_PATH, map_size=1024 * 1024 * 10)  # 10 MB

# Conversation caching settings
CONVERSATION_CACHE_SIZE = int(os.environ.get("CONVERSATION_CACHE_SIZE", 100))
CONVERSATIONS_DB_PATH = "conversations_db"
if not os.path.exists(CONVERSATIONS_DB_PATH):
    os.makedirs(CONVERSATIONS_DB_PATH)
conversations_env = lmdb.open(CONVERSATIONS_DB_PATH, map_size=1024 * 1024 * 100)  # 100 MB
conversation_keys = []

# Print debug info at startup
if not SECURE_1PSID or not SECURE_1PSIDTS:
	logger.warning("⚠️ Gemini API credentials are not set or empty! Please check your environment variables.")
	logger.warning("Make sure SECURE_1PSID and SECURE_1PSIDTS are correctly set in your .env file or environment.")
	logger.warning("If using Docker, ensure the .env file is correctly mounted and formatted.")
	logger.warning("Example format in .env file (no quotes):")
	logger.warning("SECURE_1PSID=your_secure_1psid_value_here")
	logger.warning("SECURE_1PSIDTS=your_secure_1psidts_value_here")
else:
	# Only log the first few characters for security
	logger.info(f"Credentials found. SECURE_1PSID starts with: {SECURE_1PSID[:5]}...")
	logger.info(f"Credentials found. SECURE_1PSIDTS starts with: {SECURE_1PSIDTS[:5]}...")

if not API_KEY:
	logger.warning("⚠️ API_KEY is not set or empty! API authentication will not work.")
	logger.warning("Make sure API_KEY is correctly set in your .env file or environment.")
else:
	logger.info(f"API_KEY found. API_KEY starts with: {API_KEY[:5]}...")


def correct_markdown(md_text: str) -> str:
	"""
	修正Markdown文本，移除Google搜索链接包装器，并根据显示文本简化目标URL。
	"""

	def simplify_link_target(text_content: str) -> str:
		match_colon_num = re.match(r"([^:]+:\d+)", text_content)
		if match_colon_num:
			return match_colon_num.group(1)
		return text_content

	def replacer(match: re.Match) -> str:
		outer_open_paren = match.group(1)
		display_text = match.group(2)

		new_target_url = simplify_link_target(display_text)
		new_link_segment = f"[`{display_text}`]({new_target_url})"

		if outer_open_paren:
			return f"{outer_open_paren}{new_link_segment})"
		else:
			return new_link_segment

	pattern = r"(\()?\[`([^`]+?)`\]\((https://www.google.com/search\?q=)(.*?)(?<!\\)\)\)*(\))?"

	fixed_google_links = re.sub(pattern, replacer, md_text)
	# fix wrapped markdownlink
	pattern = r"`(\[[^\]]+\]\([^\)]+\))`"
	return re.sub(pattern, r"\1", fixed_google_links)


# Pydantic models for API requests and responses
class ContentItem(BaseModel):
	type: str
	text: Optional[str] = None
	image_url: Optional[Dict[str, str]] = None


class Message(BaseModel):
	role: str
	content: Union[str, List[ContentItem]]
	name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
	model: str
	messages: List[Message]
	temperature: Optional[float] = 0.7
	top_p: Optional[float] = 1.0
	n: Optional[int] = 1
	stream: Optional[bool] = False
	max_tokens: Optional[int] = None
	presence_penalty: Optional[float] = 0
	frequency_penalty: Optional[float] = 0
	user: Optional[str] = None
	system_prompt: Optional[str] = None


class Choice(BaseModel):
	index: int
	message: Message
	finish_reason: str


class Usage(BaseModel):
	prompt_tokens: int
	completion_tokens: int
	total_tokens: int


class ChatCompletionResponse(BaseModel):
	id: str
	object: str = "chat.completion"
	created: int
	model: str
	choices: List[Choice]
	usage: Usage


class ModelData(BaseModel):
	id: str
	object: str = "model"
	created: int
	owned_by: str = "google"


class ModelList(BaseModel):
	object: str = "list"
	data: List[ModelData]


# Authentication dependency
async def verify_api_key(authorization: str = Header(None)):
	if not API_KEY:
		# If API_KEY is not set in environment, skip validation (for development)
		logger.warning("API key validation skipped - no API_KEY set in environment")
		return

	if not authorization:
		raise HTTPException(status_code=401, detail="Missing Authorization header")

	try:
		scheme, token = authorization.split()
		if scheme.lower() != "bearer":
			raise HTTPException(status_code=401, detail="Invalid authentication scheme. Use Bearer token")

		if token != API_KEY:
			raise HTTPException(status_code=401, detail="Invalid API key")
	except ValueError:
		raise HTTPException(status_code=401, detail="Invalid authorization format. Use 'Bearer YOUR_API_KEY'")

	return token


# Simple error handler middleware
@app.middleware("http")
async def error_handling(request: Request, call_next):
	try:
		return await call_next(request)
	except Exception as e:
		logger.error(f"Request failed: {str(e)}")
		return JSONResponse(status_code=500, content={"error": {"message": str(e), "type": "internal_server_error"}})


# Get list of available models
@app.get("/v1/models")
async def list_models():
	"""返回 gemini_webapi 中声明的模型列表"""
	now = int(datetime.now(tz=timezone.utc).timestamp())
	data = [
		{
			"id": m.model_name,  # 如 "gemini-2.0-flash"
			"object": "model",
			"created": now,
			"owned_by": "google-gemini-web",
		}
		for m in Model
	]
	print(data)
	return {"object": "list", "data": data}


# Helper to convert between Gemini and OpenAI model names
def map_model_name(openai_model_name: str) -> Model:
	"""根据模型名称字符串查找匹配的 Model 枚举值"""
	# 打印所有可用模型以便调试
	all_models = [m.model_name if hasattr(m, "model_name") else str(m) for m in Model]
	logger.info(f"Available models: {all_models}")

	# 如果找不到匹配项，使用默认映射
	model_keywords = {
		"gemini-pro": ["pro", "2.0"],
		"gemini-pro-vision": ["vision", "pro"],
		"gemini-flash": ["flash", "2.0"],
		"gemini-1.5-pro": ["1.5", "pro"],
		"gemini-1.5-flash": ["1.5", "flash"],
	}

	# 根据关键词匹配
	keywords = model_keywords.get(openai_model_name, [openai_model_name])

	for m in Model:
		model_name = m.model_name if hasattr(m, "model_name") else str(m)
		if all(kw.lower() in model_name.lower() for kw in keywords):
			return m

	# 如果还是找不到，返回第一个模型
	return next(iter(Model))


def get_text_from_message(message: Message) -> str:
    if isinstance(message.content, str):
        return message.content
    text = ""
    if isinstance(message.content, list):
        for item in message.content:
            if item.type == "text":
                text += item.text or ""
    return text

def prepare_files(messages: List[Message]) -> List[str]:
    temp_files = []
    for msg in messages:
        if isinstance(msg.content, list):
            for item in msg.content:
                if item.type == "image_url" and item.image_url:
                    image_url = item.image_url.get("url", "")
                    if image_url.startswith("data:image/"):
                        try:
                            base64_data = image_url.split(",")[1]
                            image_data = base64.b64decode(base64_data)
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                                tmp.write(image_data)
                                temp_files.append(tmp.name)
                        except Exception as e:
                            logger.error(f"Error processing base64 image: {str(e)}")
    return temp_files


# Helper to create gem from system prompt
async def create_gem_from_system_prompt(system_prompt: str, client: GeminiClient) -> Optional[Gem]:
	"""从系统提示词创建临时 Gem"""
	if not system_prompt:
		return None

	try:
		# Create a temporary gem with a unique ID
		gem_name = f"system-prompt-{uuid.uuid4().hex[:8]}"
		gem = await client.create_gem(
			name=gem_name,
			prompt=system_prompt,
			description="Temporary gem created from system prompt"
		)
		logger.info(f"Created temporary gem from system prompt: {gem.id}")
		return gem
	except Exception as e:
		logger.error(f"Failed to create gem from system prompt: {str(e)}")
		return None


# Dependency to get the initialized Gemini client
async def get_gemini_client():
	global gemini_client
	if gemini_client is None:
		try:
			gemini_client = GeminiClient(SECURE_1PSID, SECURE_1PSIDTS)
			await gemini_client.init(timeout=300)
		except Exception as e:
			logger.error(f"Failed to initialize Gemini client: {str(e)}")
			raise HTTPException(status_code=500, detail=f"Failed to initialize Gemini client: {str(e)}")
	return gemini_client



async def handle_conversation_flow(
    gemini_client: GeminiClient,
    model: Model,
    gem_obj: Optional[Gem],
    messages: List[Message],
):
    """处理新的或恢复的对话，并发送消息"""
    history_messages = [m for m in messages if m.role in ["user", "assistant"]]
    chat = None
    response = None

    if len(history_messages) > 1:
        history_to_check = history_messages[:-1]
        history_hash = hashlib.sha256(
            json.dumps([m.model_dump_json() for m in history_to_check]).encode("utf-8")
        ).hexdigest()

        with conversations_env.begin() as txn:
            metadata_json = txn.get(history_hash.encode("utf-8"))
            if metadata_json:
                metadata = json.loads(metadata_json.decode("utf-8"))
                chat = gemini_client.start_chat(metadata=metadata)
                message_to_send = history_messages[-1]
                files = prepare_files([message_to_send])
                response = await chat.send_message(
                    get_text_from_message(message_to_send), files=files
                )
                logger.info(f"Resuming conversation from cache with hash: {history_hash}")

    if not chat:
        chat = gemini_client.start_chat(model=model, gem=gem_obj)
        logger.info("Starting a new conversation.")
        for message in history_messages:
            files = prepare_files([message])
            response = await chat.send_message(
                get_text_from_message(message), files=files
            )

    return chat, response, history_messages

@app.post("/v1/chat/completions")
async def create_chat_completion(
    request: ChatCompletionRequest,
    gemini_client: GeminiClient = Depends(get_gemini_client),
    api_key: str = Depends(verify_api_key),
):
    try:
        messages = request.messages

        # Extract system prompt and handle gem caching
        system_prompt_msg = next((m for m in messages if m.role == "system"), None)
        final_system_prompt = request.system_prompt or (
            get_text_from_message(system_prompt_msg) if system_prompt_msg else None
        )

        gem_obj = None
        if final_system_prompt:
            with gems_env.begin(write=True) as txn:
                gem_json_bytes = txn.get(final_system_prompt.encode("utf-8"))
                if gem_json_bytes:
                    gem_data = json.loads(gem_json_bytes.decode("utf-8"))
                    gem_obj = Gem(**gem_data)
                    logger.info(f"Using cached gem with ID: {gem_obj.id}")
                else:
                    gem_obj = await create_gem_from_system_prompt(
                        final_system_prompt, gemini_client
                    )
                    if gem_obj:
                        gem_json = gem_obj.model_dump_json()
                        txn.put(
                            final_system_prompt.encode("utf-8"),
                            gem_json.encode("utf-8"),
                        )
                        logger.info(f"Created and cached new gem with ID: {gem_obj.id}")

        model = map_model_name(request.model)
        chat, response, history_messages = await handle_conversation_flow(
            gemini_client, model, gem_obj, messages
        )

        reply_text = response.text if response else ""
        reply_text = correct_markdown(reply_text)

        with conversations_env.begin(write=True) as txn:
            new_history = history_messages + [
                Message(role="assistant", content=reply_text)
            ]
            new_history_hash = hashlib.sha256(
                json.dumps([m.model_dump_json() for m in new_history]).encode("utf-8")
            ).hexdigest()

            metadata_to_store = chat.metadata
            txn.put(
                new_history_hash.encode("utf-8"),
                json.dumps(metadata_to_store).encode("utf-8"),
            )

            if new_history_hash not in conversation_keys:
                conversation_keys.append(new_history_hash)
                if len(conversation_keys) > CONVERSATION_CACHE_SIZE:
                    oldest_key = conversation_keys.pop(0)
                    txn.delete(oldest_key.encode("utf-8"))
                    logger.info(f"Cache full. Evicted oldest conversation: {oldest_key}")
            logger.info(f"Stored new conversation state with hash: {new_history_hash}")

        completion_id = f"chatcmpl-{uuid.uuid4()}"
        created_time = int(time.time())

        if request.stream:

            async def generate_stream():
                data = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created_time,
                    "model": request.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant"},
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(data)}\n\n"

                for char in reply_text:
                    data = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created_time,
                        "model": request.model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": char},
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(data)}\n\n"
                    await asyncio.sleep(0.01)

                data = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created_time,
                    "model": request.model,
                    "choices": [
                        {"index": 0, "delta": {}, "finish_reason": "stop"}
                    ],
                }
                yield f"data: {json.dumps(data)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(generate_stream(), media_type="text/event-stream")
        else:
            result = {
                "id": completion_id,
                "object": "chat.completion",
                "created": created_time,
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": reply_text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 0,  # This is now hard to calculate
                    "completion_tokens": len(reply_text.split()),
                    "total_tokens": len(reply_text.split()),
                },
            }
            logger.info(f"Returning response: {result}")
            return result

    except Exception as e:
        logger.error(f"Error generating completion: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Error generating completion: {str(e)}"
        )


@app.get("/")
async def root():
	return {"status": "online", "message": "Gemini API FastAPI Server is running"}


if __name__ == "__main__":
	import uvicorn

	uvicorn.run("main:app", host="0.0.0.0", port=8000, log_level="info")
