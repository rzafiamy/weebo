import time
import logging
import tiktoken  # Ensure you have tiktoken installed: pip install tiktoken
from ollama import chat
import config

# Configure logging to capture all levels
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("chatbot.log")
    ]
)

class ChatBot:
    """
    A chatbot class that uses Ollama's API for generating responses with token limit management.
    """
    
    def __init__(self, rag=None):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Initializing ChatBot...")
        self.messages = []  # to store conversation history
        self.rag = rag

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
        self.token_limit = getattr(config, "TOKEN_LIMIT", 4092)  # Set a default max token limit

        # Initialize tokenizer
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def _get_token_count(self, messages):
        """
        Calculates the total token count of the provided messages.
        """
        text = " ".join([msg['content'] for msg in messages])
        return len(self.tokenizer.encode(text))

    def _manage_token_limit(self):
        """
        Checks if token count exceeds the limit and resets if necessary.
        """
        token_count = self._get_token_count(self.messages)
        self.logger.info(f"Current token count: {token_count}")
        if token_count > self.token_limit:
            self.logger.warning("Token limit exceeded! Resetting conversation history.")
            self.messages = []  # Reset conversation history
    
    def _call_chat_api(self, prompt: str) -> str:
        """
        Calls the Ollama chat API with a prompt and returns the response.
        """
        self.logger.debug("Calling Ollama API...")
        messages = [{'role': 'system', 'content': self.system_prompt}]
        messages.extend(self.messages)
        messages.append({'role': 'user', 'content': prompt})

        self.logger.debug(f"Full messages payload: {messages}")
        stream = chat(model=self.model, messages=messages, stream=True)
        response_chunks = []
        for chunk in stream:
            try:
                content = chunk.get('message', {}).get('content', "")
                response_chunks.append(content)
                self.logger.debug(f"Received chunk: {content}")
            except Exception as e:
                self.logger.exception("Error processing a chunk from the API.")
        return "".join(response_chunks)

    def generate_response(self, prompt: str) -> str:
        """
        Generates a response from the chatbot for a given user prompt.
        """
        self.logger.debug(f"Received prompt: {prompt}")
        self._manage_token_limit()  # Check and reset context if needed

        if self.rag:
            # Query rag for the prompt
            retrieved = self.rag.query(prompt)
            logging.info("--------------")
            logging.info(f"RAG: {retrieved[0]}")
            logging.info("--------------")

            prompt = "\n\n".join(retrieved) + "\n\n" + prompt

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
                time.sleep(self.retry_delay * (2 ** (attempt - 1)))  # Exponential backoff

        if not response.strip():
            self.logger.error("Failed to generate a valid response after retries. Returning fallback reply.")
            response = self.no_voice_reply

        self.messages.append({'role': 'assistant', 'content': response})
        return response

    def add_context(self, role: str, content: str) -> None:
        """
        Adds a new message to the conversation history and checks token limit.
        """
        if role not in ('user', 'assistant'):
            self.logger.error("Invalid role specified for context message.")
            raise ValueError("Role must be either 'user' or 'assistant'.")
        
        self.messages.append({'role': role, 'content': content})
        self._manage_token_limit()
        self.logger.debug(f"Added context message: {role} -> {content}")

# Example usage:
if __name__ == "__main__":
    bot = ChatBot()
    user_prompt = "Hello, how can you help me today?"
    print(bot.generate_response(user_prompt))
