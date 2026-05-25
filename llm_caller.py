import os
import sys
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

import openai
from google import genai
from google.genai import types

class LLMCaller:

    def __init__(
        self,
        api_path=None,
        provider="openrouter",
        model="openai/gpt-4o-mini"
        ):

        # read api keys from file or default path
        try:
            self.api_path = "configs/llm_api.json" if api_path is None else api_path
            with open(self.api_path, "r") as f:
                self.api_keys = json.load(f)
            self.openrouter_api_key = self.api_keys.get("openrouter_api") or os.getenv("OPENROUTER_API_KEY")
            self.openai_api_key = self.api_keys.get("openai_api") or os.getenv("OPENAI_API_KEY")
            self.google_api_key = self.api_keys.get("google_api") or os.getenv("GOOGLE_API_KEY")
            self.deepseek_api_key = self.api_keys.get("deepseek_api") or os.getenv("DEEPSEEK_API_KEY")
        except FileNotFoundError:
            raise FileNotFoundError(f"API key file not found at {self.api_path}")
        
        # set provider and model
        self.provider, self.model = provider, model
        if self.provider != "openrouter" and "/" in model:
            self.model = model.split("/", 1)[1]
    
    def call(self, input, temperature=1.0):
        """
        inputs = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ]
        """

        if self.provider == "openrouter":
            return self.call_openrouter(input, temperature)
        elif self.provider == "openai":
            return self.call_openai(input, temperature)
        elif self.provider == "google":
            return self.call_google(input, temperature)
        elif self.provider == "deepseek":
            return self.call_deepseek(input, temperature)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def call_batch(self, inputs, max_workers=8, temperature=1.0):
        """
        inputs = [
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"},
            ],
            [
                {"role": "system", "content": "You are a knowledgeable guide."},
                {"role": "user", "content": "How are you?"},
            ],
            [
                {"role": "system", "content": "You are a witty comedian."},
                {"role": "user", "content": "Tell me a joke."},
            ]
        ]  # len(inputs) = n -> n responses
        """

        results = [None] * len(inputs)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {executor.submit(self.call, inp, temperature): i for i, inp in enumerate(inputs)}
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                results[idx] = future.result()
        return results


    def call_openai(self, input, temperature=1.0):

        for attempt in range(3):
            try:
                client = openai.OpenAI(api_key=self.openai_api_key)
                response = client.responses.create(
                    model=self.model,
                    input=input,
                    temperature=temperature,
                )
                return response.output_text
            except Exception as e:
                if attempt < 2:
                    print(f"Error calling OpenAI API (attempt {attempt + 1}/3): {e}")
                    continue
                else:
                    print(f"Error calling OpenAI API (attempt {attempt + 1}/3): {e}")
                    sys.exit(1)

    def call_openrouter(self, input, temperature=1.0):

        for attempt in range(3):
            try:
                client = openai.OpenAI(api_key=self.openrouter_api_key, base_url="https://openrouter.ai/api/v1")
                response = client.responses.create(
                    model=self.model,
                    input=input,
                    temperature=temperature
                )
                return response.output_text
            except Exception as e:
                if attempt < 2:
                    print(f"Error calling OpenRouter API (attempt {attempt + 1}/3): {e}")
                    continue
                else:
                    print(f"Error calling OpenRouter API (attempt {attempt + 1}/3): {e}")
                    sys.exit(1)

    def call_google(self, input, temperature=1.0):

        system_instruction, user_content = None, None
        for item in input:
            if item["role"] == "system":
                system_instruction = item["content"]
            if item["role"] == "user":
                user_content = item["content"]

        for attempt in range(3):
            try:
                client = genai.Client(api_key=self.google_api_key)
                response = client.models.generate_content(
                    model=self.model.split("/", 1)[1],
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=temperature,
                    ) if system_instruction else types.GenerateContentConfig(temperature=temperature),
                    contents=user_content,
                )
                return response.text
            except Exception as e:
                if attempt < 2:
                    print(f"Error calling Google API (attempt {attempt + 1}/3): {e}")
                    continue
                else:
                    print(f"Error calling Google API (attempt {attempt + 1}/3): {e}")
                    sys.exit(1)

    def call_deepseek(self, input, temperature=1.0):

        for attempt in range(3):
            try:
                client = openai.OpenAI(
                    api_key=self.deepseek_api_key,
                    base_url="https://api.deepseek.com"
                )
                response = client.chat.completions.create(
                    model=self.model,
                    messages=input,
                    temperature=temperature,
                )
                return response.choices[0].message.content
            except Exception as e:
                if attempt < 2:
                    print(f"Error calling DeepSeek API (attempt {attempt + 1}/3): {e}")
                    continue
                else:
                    print(f"Error calling DeepSeek API (attempt {attempt + 1}/3): {e}")
                    sys.exit(1)
    
    # print the LLMCaller info
    def __repr__(self):
        return (
            f"LLMCaller(\n"
            f"  provider={self.provider},\n"
            f"  model={self.model}\n"
            f")"
        )
