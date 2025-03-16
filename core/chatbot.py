from ollama import chat
import config
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class ChatBot:
    def __init__(self):
        logging.info("Initializing ChatBot...")
        self.messages = []
        self.system_prompt = config.SYSTEM_PROMPT

    def generate_response(self, prompt: str) -> str:
        stream = chat(model='gemma3:1B', messages=[{'role': 'system', 'content': self.system_prompt}, {'role': 'user', 'content': prompt}] + self.messages, stream=True)
        response = "".join(chunk['message']['content'] for chunk in stream)
        self.messages.append({'role': 'assistant', 'content': response})
        return response