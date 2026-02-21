"""Test the fine-tuned ReachyMini voice SLM."""

import json
import os
import time

from mlx_lm import load, generate

_HERE = os.path.dirname(os.path.abspath(__file__))

# Load tools from job_description.json
with open(os.path.join(_HERE, "data", "job_description.json")) as f:
    JOB_DESC = json.load(f)
TOOLS = JOB_DESC["tools"]

# Test inputs and expected function names
TESTS = [
    ("Hey Tsubasa", "greeting"),
    ("おはよう", "greeting"),
    ("Play some jazz", "play_music"),
    ("踊って", "dance"),
    ("What time is it", "check_time"),
    ("音楽止めて", "stop_music"),
    ("Turn up the volume", "volume"),
    ("What is the meaning of life", "intent_unclear"),
    ("Say hello to Ray", "speak"),
    ("笑って", "set_emotion"),
    ("Nod your head", "nod"),
    ("首振って", "shake_head"),
    ("Look around", "look_around"),
    ("おやすみ", "goodbye"),
    ("ありがとう", "thank_you"),
    ("Set volume to 70", "volume"),
    ("ポケモンの曲かけて", "play_music"),
    ("Help me with homework", "intent_unclear"),
    ("Uh play uh some chill music", "play_music"),
    ("こっち見て", "look_around"),
]


def build_prompt(tokenizer, user_text: str) -> str:
    """Build a prompt with tools for the model."""
    messages = [{"role": "user", "content": user_text}]
    return tokenizer.apply_chat_template(
        messages,
        tools=TOOLS,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False,
    )


def extract_function_call(response: str) -> dict | None:
    """Try to extract a function call from the model response."""
    # Look for JSON in the response
    response = response.strip()

    # Try to find tool_call pattern
    # Qwen3 tool calling format: <tool_call>{"name": ..., "arguments": ...}</tool_call>
    if "<tool_call>" in response:
        start = response.index("<tool_call>") + len("<tool_call>")
        end = response.index("</tool_call>") if "</tool_call>" in response else len(response)
        json_str = response[start:end].strip()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

    # Try direct JSON parse
    try:
        parsed = json.loads(response)
        if "name" in parsed:
            return parsed
    except json.JSONDecodeError:
        pass

    # Try to find JSON object in response
    for i, ch in enumerate(response):
        if ch == "{":
            for j in range(len(response) - 1, i, -1):
                if response[j] == "}":
                    try:
                        parsed = json.loads(response[i : j + 1])
                        if "name" in parsed:
                            return parsed
                    except json.JSONDecodeError:
                        continue

    return None


def main():
    model_path = os.path.join(_HERE, "fused_model")
    print(f"Loading model from {model_path}...")
    model, tokenizer = load(model_path)
    print("Model loaded!\n")

    correct = 0
    total = len(TESTS)
    times = []

    for user_text, expected in TESTS:
        prompt = build_prompt(tokenizer, user_text)

        t0 = time.perf_counter()
        response = generate(model, tokenizer, prompt=prompt, max_tokens=200)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)

        func_call = extract_function_call(response)
        got_name = func_call["name"] if func_call else "PARSE_ERROR"
        ok = got_name == expected

        if ok:
            correct += 1

        status = "✓" if ok else "✗"
        args_str = ""
        if func_call and func_call.get("arguments"):
            args_str = f" {func_call['arguments']}"
        print(f"  {status} [{elapsed*1000:.0f}ms] \"{user_text}\" → {got_name}{args_str}" +
              (f" (expected {expected})" if not ok else ""))

    avg_ms = sum(times) / len(times) * 1000
    print(f"\n{'='*60}")
    print(f"Accuracy: {correct}/{total} ({correct/total:.1%})")
    print(f"Avg latency: {avg_ms:.0f}ms")
    print(f"Min/Max: {min(times)*1000:.0f}ms / {max(times)*1000:.0f}ms")


if __name__ == "__main__":
    main()
