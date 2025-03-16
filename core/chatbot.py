from ollama import chat
import config
import logging

# Configure logging to capture all levels
logging.basicConfig(
    level=logging.DEBUG,  # Change to DEBUG to capture all logs
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Logs to console
        logging.FileHandler("chatbot.log")  # Logs to file
    ]
)

class ChatBot:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Initializing ChatBot...")
        self.messages = []
        try:
            self.system_prompt = config.SYSTEM_PROMPT
        except AttributeError:
            self.logger.error("SYSTEM_PROMPT is not defined in config.")
            self.system_prompt = "Default system prompt."

    def generate_response(self, prompt: str) -> str:
        self.logger.debug(f"Received prompt: {prompt}")
        try:
            stream = chat(
                model='gemma3:1B',
                messages=[
                    {'role': 'system', 'content': self.system_prompt},
                    {'role': 'user', 'content': prompt}
                ] + self.messages,
                stream=True
            )
            response = "".join(chunk['message']['content'] for chunk in stream)
            self.logger.info("Generated response successfully.")
        except Exception as e:
            self.logger.exception("Error occurred while generating response.")
            response = "An error occurred. Please try again later."
        
        if not response:
            response = "I'm sorry, I don't understand what you're asking."
        
        self.messages.append({'role': 'assistant', 'content': response})
        return response
