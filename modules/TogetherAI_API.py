from typing import List, Dict
from together import Together
import time, json, os, requests


keys = []       # Add your togetherAI API keys here
assert os.path.exists("keys/mine.txt")
with open("keys/mine.txt") as f:
    keys = f.read().splitlines()
key_cnt = 0


def _get_key():
    global key_cnt
    key = keys[key_cnt]
    key_cnt = (key_cnt + 1) % len(keys)
    return key


def text_completion(prompt, model_ckpt, max_tokens=256, temperature=0.8, top_k=40, top_p=0.95, repetition_penalty=1, stop=None):
    while True:
        try:
            client = Together(api_key=_get_key())
            response = client.completions.create(
                model=model_ckpt,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                stop=stop,
            )
            break
        except:
            print("Together AI API failed. Retrying...")
            
    return response.choices[0].text


def chat_completion(prompt, model_ckpt, system_prompt: str = "You are a helpful AI assistant.", max_tokens=256, temperature=0.8, top_k=40, top_p=0.95, repetition_penalty=1, stop=None):
    while True:
        client = Together(api_key=_get_key())
        try:
            response = client.chat.completions.create(
                model=model_ckpt,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                stop=stop,
            )
            break
        except:
            print("Together AI API failed. Retrying...")

    return response.choices[0].message.content


def _test01():
    prompt = "What are some fun things to do in New York"
    model_ckpt = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    model_output = chat_completion(prompt, model_ckpt)
    print(model_output)


if __name__ == "__main__":
    _test01()
