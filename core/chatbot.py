import time
import logging
from ollama import chat
import config

# Configure logging to capture all levels
logging.basicConfig(
    level=logging.INFO,  # Set to DEBUG to capture all logs
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.StreamHandler(),   # Logs to console
        logging.FileHandler("chatbot.log")  # Logs to file
    ]
)

class ChatBot:
    """
    A chatbot class that uses Ollama's API for generating responses.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Initializing ChatBot...")
        self.messages = []  # to store conversation history

        # Load system prompt from config; fallback to default if not provided.
        self.system_prompt = getattr(config, "SYSTEM_PROMPT", "Default system prompt.")
        if self.system_prompt == "Default system prompt.":
            self.logger.warning("SYSTEM_PROMPT not defined in config. Using default prompt.")

        # Fallback message in case of errors
        self.no_voice_reply = getattr(config, "NO_VOICE_REPLY", "Sorry, I am unable to generate a reply at this moment.")

        # Define model and retry configuration from config if available.
        self.model = getattr(config, "MODEL", "gemma3:1B")
        self.max_retries = getattr(config, "MAX_RETRIES", 3)
        self.retry_delay = getattr(config, "RETRY_DELAY", 1)  # in seconds

    def _call_chat_api(self, prompt: str) -> str:
        """
        Calls the Ollama chat API with a prompt and returns the response.
        Uses streaming and aggregates all chunks into a full response.
        """
        self.logger.debug("Calling Ollama API...")
        # Build the message list with system, conversation context, and the current user prompt
        messages = [{'role': 'system', 'content': self.system_prompt}]
        messages.extend(self.messages)
        messages.append({'role': 'user', 'content': prompt})

        self.logger.debug(f"Full messages payload: {messages}")
        stream = chat(
            model=self.model,
            messages=messages,
            stream=True
        )
        response_chunks = []
        for chunk in stream:
            try:
                # Ensure that each chunk has the expected structure
                content = chunk.get('message', {}).get('content', "")
                response_chunks.append(content)
                self.logger.debug(f"Received chunk: {content}")
            except Exception as e:
                self.logger.exception("Error processing a chunk from the API.")
        return "".join(response_chunks)

    def generate_response(self, prompt: str) -> str:
        """
        Generates a response from the chatbot for a given user prompt.
        In case of failure, retries with exponential backoff.
        """
        self.logger.debug(f"Received prompt: {prompt}")
        attempt = 0
        response = ""
        while attempt < self.max_retries:
            try:
                response = self._call_chat_api(prompt)
                if response.strip():
                    self.logger.info("Generated response successfully.")
                    break
                else:
                    raise ValueError("Empty response received from API.")
            except Exception as e:
                attempt += 1
                self.logger.exception(f"Attempt {attempt}: Error occurred while generating response.")
                time.sleep(self.retry_delay * (2 ** (attempt - 1)))  # exponential backoff

        if not response.strip():
            self.logger.error("Failed to generate a valid response after retries. Returning fallback reply.")
            response = self.no_voice_reply

        # Append assistant's response to the conversation history
        self.messages.append({'role': 'assistant', 'content': response})
        return response

    def add_context(self, role: str, content: str) -> None:
        """
        Adds a new message to the conversation history.
        
        Parameters:
            role (str): The role of the message sender ('user' or 'assistant').
            content (str): The content of the message.
        """
        if role not in ('user', 'assistant'):
            self.logger.error("Invalid role specified for context message.")
            raise ValueError("Role must be either 'user' or 'assistant'.")
        self.messages.append({'role': role, 'content': content})
        self.logger.debug(f"Added context message: {role} -> {content}")

# Example usage:
if __name__ == "__main__":
    bot = ChatBot()
    user_prompt = "Hello, how can you help me today?"
    print(bot.generate_response(user_prompt))
