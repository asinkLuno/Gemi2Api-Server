import json

import requests

# Get API_KEY from environment variable.
# You can run this script with: API_KEY=your_api_key_here python test_chat.py
API_KEY = "0919"


URL = "http://localhost:8123/v1/chat/completions"
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}


def run_test():
	# Initial message history
	messages = [
		{"role": "user", "content": "Hi there! What's the capital of France?"},
	]

	print("--- First round ---")
	print(f"User: {messages[0]['content']}")

	data = {
		"model": "gemini-2.5-flash",
		"messages": messages,
		"system_prompt": "You are a helpful assistant.",
	}

	try:
		# First request
		response = requests.post(URL, headers=HEADERS, json=data)
		response.raise_for_status()

		response_data = response.json()
		assistant_message = response_data["choices"][0]["message"]
		print(f"Assistant: {assistant_message['content']}")

		# Add assistant's response to history
		messages.append(assistant_message)

		# --- Second round ---
		print("\n--- Second round ---")
		second_user_message = {"role": "user", "content": "And what is its population?"}
		messages.append(second_user_message)
		print(f"User: {second_user_message['content']}")

		data["messages"] = messages

		# Second request
		response = requests.post(URL, headers=HEADERS, json=data)
		response.raise_for_status()

		response_data = response.json()
		assistant_message_2 = response_data["choices"][0]["message"]
		print(f"Assistant: {assistant_message_2['content']}")

	except requests.exceptions.RequestException as e:
		print(f"An error occurred: {e}")
	except json.JSONDecodeError:
		print(f"Failed to decode JSON from response: {response.text}")


if __name__ == "__main__":
	run_test()
