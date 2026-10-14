import config
from openai import OpenAI

GROQURL = "https://api.groq.com/openai/v1"
MODELS = getattr(config, "GROQ_MODELS", ["llama-3.1-8b-instant", "mixtral-8x7b-32768"])

def generate_response(prompt: str, temperature: float = 0.3, max_tokens: int = 512) -> str:
    key = getattr(config, "GROQ_API_KEY", None)
    if not key:
        return "Error: groq api key is  missing in config.py"
    c = OpenAI(api_key=key, base_url=GROQURL)

    last_err = None
    for m in MODELS:
        try:
            r = c.chat.completions.create(
                model=m,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return r.choices[0].message.content
        except Exception as e:
            last_err = e

    return (
        "model failed"
        f"Tried models: {MODELS}"
        "Fix:"
        "1) Switch to hf by importing hf.py in main.py or config.py"
        "2) Replace Groq model in groq.py (GROQ_MODELS)"
        f"Details: {type(last_err).__name__}: {last_err}"
    )