"""ReachyMini Voice Orchestrator — maps SLM tool calls to robot commands.

Takes function calls from a fine-tuned SLM and translates them into
/tmp/reachy_command format for reachy_hub.py.

Architecture:
    Whisper (STT) → SLM (intent + slots) → Orchestrator → reachy_command → reachy_hub.py
    For intent_unclear: escalate to Claude for complex reasoning/conversation.

Usage:
    python orchestrator.py --model model --port 8000 [--debug]
"""

from __future__ import annotations

import argparse
import datetime
import json
import os

from openai import OpenAI

# ---------------------------------------------------------------------------
# Tools definition — loaded from job_description.json
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_JOB_DESC_PATH = os.path.join(_HERE, "data", "job_description.json")

with open(_JOB_DESC_PATH) as f:
    _JOB_DESC = json.load(f)

TOOLS: list[dict] = _JOB_DESC["tools"]
TASK_DESCRIPTION: str = _JOB_DESC["task_description"]

# ---------------------------------------------------------------------------
# Command mapping: SLM function name → /tmp/reachy_command format
# ---------------------------------------------------------------------------
COMMAND_FILE = "/tmp/reachy_command"

# Emotion enum → reachy_hub emotion name
EMOTION_MAP: dict[str, str] = {
    "happy": "cheerful1",
    "sad": "sad1",
    "angry": "angry1",
    "surprised": "surprised1",
    "thinking": "thinking",
    "neutral": "neutral",
    "sleepy": "sleepy1",
}

# Dance style → reachy_hub dance name
DANCE_MAP: dict[str, str] = {
    "happy": "groovy_sway_and_roll",
    "energetic": "groovy_sway_and_roll",
    "gentle": "gentle_sway_and_roll",
}

# Volume step for relative changes
VOLUME_STEP = 15
DEFAULT_VOLUME = 50

# Bilingual response templates
RESPONSES = {
    "greeting": {
        "en": "Hey! What's up?",
        "ja": "おはよ！何する？",
    },
    "goodbye": {
        "en": "See you later!",
        "ja": "おやすみ！",
    },
    "thank_you": {
        "en": "You're welcome!",
        "ja": "どういたしまして！",
    },
    "intent_unclear": {
        "en": "Hmm, let me think about that...",
        "ja": "うーん、ちょっと考えるね…",
    },
    "check_time": {
        "en": "It's {time} right now.",
        "ja": "今{time}だよ。",
    },
    "play_music": {
        "en": "Playing {query}!",
        "ja": "{query}かけるね！",
    },
    "play_music_default": {
        "en": "Playing some music!",
        "ja": "音楽かけるね！",
    },
    "stop_music": {
        "en": "Music stopped.",
        "ja": "音楽止めたよ。",
    },
    "volume_up": {
        "en": "Turning it up!",
        "ja": "音量上げるね！",
    },
    "volume_down": {
        "en": "Turning it down.",
        "ja": "音量下げるね。",
    },
    "volume_set": {
        "en": "Volume set to {level}.",
        "ja": "音量{level}にしたよ。",
    },
    "dance": {
        "en": "Let's dance!",
        "ja": "踊るよ！",
    },
    "look_around": {
        "en": "Let me look around...",
        "ja": "見てみるね…",
    },
}

# Motion to play with certain responses
RESPONSE_MOTIONS = {
    "greeting": "cheerful1",
    "goodbye": "sad1",
    "thank_you": "nod",
    "dance": "cheerful1",
}


# ---------------------------------------------------------------------------
# Language detection (simple heuristic)
# ---------------------------------------------------------------------------
def detect_language(text: str) -> str:
    """Detect if text is Japanese or English (simple heuristic)."""
    for ch in text:
        if "\u3040" <= ch <= "\u30ff" or "\u4e00" <= ch <= "\u9fff":
            return "ja"
    return "en"


# ---------------------------------------------------------------------------
# SLM Client — stateless wrapper around an OpenAI-compatible endpoint
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "You are a tool-calling model working on:\n"
        f"<task_description>{TASK_DESCRIPTION}</task_description>\n\n"
        "Respond to the conversation history by generating an appropriate tool call that "
        "satisfies the user request. Generate only the tool call according to the provided "
        "tool schema, do not generate anything else. Always respond with a tool call.\n\n"
    ),
}


class SLMClient:
    """Lightweight client for a llama.cpp / Ollama / vLLM server."""

    def __init__(self, model_name: str, api_key: str = "EMPTY", port: int = 8000):
        self.model_name = model_name
        self.client = OpenAI(
            base_url=f"http://127.0.0.1:{port}/v1",
            api_key=api_key,
        )

    def invoke(self, conversation_history: list[dict]) -> dict | str:
        """Send full conversation history to the SLM and return a parsed
        function-call dict ``{"name": ..., "arguments": ...}`` or an error
        string."""
        messages = [SYSTEM_PROMPT] + conversation_history

        chat_response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0,
            tools=TOOLS,
            tool_choice="required",
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        response = chat_response.choices[0].message

        # Path A: proper tool_calls
        if response.tool_calls:
            fn = response.tool_calls[0].function
            arguments = fn.arguments
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            return {"name": fn.name, "arguments": arguments}

        # Path B: JSON in content (fallback)
        if response.content:
            try:
                parsed = json.loads(response.content.strip())
                if "name" in parsed:
                    args = parsed.get("arguments", parsed.get("parameters", {}))
                    if isinstance(args, str):
                        args = json.loads(args)
                    return {"name": parsed["name"], "arguments": args}
            except (json.JSONDecodeError, KeyError):
                pass

        return f"No valid tool call: {response}"


# ---------------------------------------------------------------------------
# Orchestrator — deterministic command dispatcher
# ---------------------------------------------------------------------------
class ReachyOrchestrator:
    """Maps SLM function calls to reachy_hub.py commands."""

    def __init__(self, slm_client: SLMClient, debug: bool = False):
        self.slm = slm_client
        self.debug = debug
        self.conversation_history: list[dict] = []
        self.current_volume = DEFAULT_VOLUME

    def process_utterance(self, transcript: str) -> str | None:
        """Full turn: user text in → bot response out (+ side-effect command)."""
        # Detect language from user input
        lang = detect_language(transcript)

        # Append user turn
        self.conversation_history.append({"role": "user", "content": transcript})

        # Call SLM
        function_call = self.slm.invoke(self.conversation_history)

        if self.debug:
            print(f"  [DEBUG] SLM returned: {function_call}")

        # If SLM failed, treat as unclear
        if isinstance(function_call, str):
            self.conversation_history.append({"role": "assistant", "content": ""})
            return self._handle_unclear(lang)

        # Record assistant turn in history
        args_str = (
            json.dumps(function_call["arguments"])
            if isinstance(function_call["arguments"], dict)
            else function_call["arguments"]
        )
        self.conversation_history.append({
            "role": "assistant",
            "tool_calls": [{
                "type": "function",
                "function": {
                    "name": function_call["name"],
                    "arguments": args_str,
                },
            }],
        })

        # Dispatch
        return self._dispatch(function_call, lang)

    def reset(self) -> None:
        self.conversation_history = []

    def _dispatch(self, function_call: dict, lang: str) -> str | None:
        """Route function call to the appropriate handler."""
        name = function_call["name"]
        args = function_call.get("arguments", {})

        if name == "speak":
            return self._handle_speak(args, lang)
        elif name == "nod":
            return self._handle_nod(lang)
        elif name == "shake_head":
            return self._handle_shake(lang)
        elif name == "look_around":
            return self._handle_look(lang)
        elif name == "set_emotion":
            return self._handle_emotion(args, lang)
        elif name == "play_music":
            return self._handle_play_music(args, lang)
        elif name == "stop_music":
            return self._handle_stop_music(lang)
        elif name == "volume":
            return self._handle_volume(args, lang)
        elif name == "check_time":
            return self._handle_check_time(lang)
        elif name == "dance":
            return self._handle_dance(args, lang)
        elif name == "greeting":
            return self._handle_greeting(lang)
        elif name == "goodbye":
            return self._handle_goodbye(lang)
        elif name == "thank_you":
            return self._handle_thank_you(lang)
        elif name == "intent_unclear":
            return self._handle_unclear(lang)
        else:
            return self._handle_unclear(lang)

    # --- Individual handlers ---

    def _handle_speak(self, args: dict, lang: str) -> str:
        text = args.get("text", "")
        speak_lang = args.get("language", lang)
        motion = args.get("motion")

        if not text:
            return RESPONSES["intent_unclear"][lang]

        # Build reachy_command
        cmd = f"say:{text}"
        if motion:
            cmd = f"{cmd} + {motion}"

        self._send_command(cmd)
        return f"[speak] {text}"

    def _handle_nod(self, lang: str) -> str:
        self._send_command("nod")
        return "[nod]"

    def _handle_shake(self, lang: str) -> str:
        self._send_command("shake")
        return "[shake]"

    def _handle_look(self, lang: str) -> str:
        self._send_command("look")
        resp = RESPONSES["look_around"][lang]
        return resp

    def _handle_emotion(self, args: dict, lang: str) -> str:
        emotion = args.get("emotion", "neutral")
        reachy_emotion = EMOTION_MAP.get(emotion, emotion)
        self._send_command(reachy_emotion)
        return f"[emotion: {emotion}]"

    def _handle_play_music(self, args: dict, lang: str) -> str:
        query = args.get("query")
        if query:
            # TODO: integrate with spotify_play skill
            resp = RESPONSES["play_music"][lang].format(query=query)
        else:
            resp = RESPONSES["play_music_default"][lang]
        # For now, speak the response. Real impl would trigger Spotify.
        self._send_command(f"say:{resp} + cheerful1")
        return resp

    def _handle_stop_music(self, lang: str) -> str:
        # TODO: integrate with spotify stop
        resp = RESPONSES["stop_music"][lang]
        self._send_command(f"say:{resp}")
        return resp

    def _handle_volume(self, args: dict, lang: str) -> str:
        level = args.get("level")
        direction = args.get("direction")

        if level is not None:
            self.current_volume = max(0, min(100, int(level)))
            self._send_command(f"volume:{self.current_volume}")
            return RESPONSES["volume_set"][lang].format(level=self.current_volume)
        elif direction == "up":
            self.current_volume = min(100, self.current_volume + VOLUME_STEP)
            self._send_command(f"volume:{self.current_volume}")
            return RESPONSES["volume_up"][lang]
        elif direction == "down":
            self.current_volume = max(0, self.current_volume - VOLUME_STEP)
            self._send_command(f"volume:{self.current_volume}")
            return RESPONSES["volume_down"][lang]
        else:
            return RESPONSES["intent_unclear"][lang]

    def _handle_check_time(self, lang: str) -> str:
        now = datetime.datetime.now()
        if lang == "ja":
            time_str = now.strftime("%H時%M分")
        else:
            time_str = now.strftime("%I:%M %p")
        resp = RESPONSES["check_time"][lang].format(time=time_str)
        self._send_command(f"say:{resp} + nod")
        return resp

    def _handle_dance(self, args: dict, lang: str) -> str:
        style = args.get("style", "happy")
        dance_name = DANCE_MAP.get(style, "groovy_sway_and_roll")
        self._send_command(f"dance:{dance_name}")
        return RESPONSES["dance"][lang]

    def _handle_greeting(self, lang: str) -> str:
        resp = RESPONSES["greeting"][lang]
        motion = RESPONSE_MOTIONS["greeting"]
        self._send_command(f"say:{resp} + {motion}")
        return resp

    def _handle_goodbye(self, lang: str) -> str:
        resp = RESPONSES["goodbye"][lang]
        self._send_command(f"say:{resp}")
        return resp

    def _handle_thank_you(self, lang: str) -> str:
        resp = RESPONSES["thank_you"][lang]
        motion = RESPONSE_MOTIONS["thank_you"]
        self._send_command(f"say:{resp} + {motion}")
        return resp

    def _handle_unclear(self, lang: str) -> str:
        """Escalate to full LLM (Claude) for complex requests."""
        resp = RESPONSES["intent_unclear"][lang]
        self._send_command(f"say:{resp} + thinking")
        # TODO: forward last user message to Claude via chat mode
        return resp

    # --- Command I/O ---

    def _send_command(self, command: str) -> None:
        """Write command to /tmp/reachy_command for reachy_hub.py."""
        if self.debug:
            print(f"  [CMD] {command}")
        with open(COMMAND_FILE, "w") as f:
            f.write(command)


# ---------------------------------------------------------------------------
# Evaluation helper — run test.jsonl through the SLM and score accuracy
# ---------------------------------------------------------------------------
def evaluate(slm: SLMClient, test_path: str, debug: bool = False) -> dict:
    """Run held-out test examples through the SLM and compute accuracy."""
    correct = 0
    total = 0
    errors: list[dict] = []

    with open(test_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            example = json.loads(line)
            conversation = json.loads(example["question"])
            expected = json.loads(example["answer"])

            result = slm.invoke(conversation)

            total += 1

            if isinstance(result, str):
                errors.append({
                    "input": conversation[-1]["content"],
                    "expected": expected,
                    "got": result,
                })
                continue

            # Compare function name
            got_name = result["name"]
            exp_name = expected["name"]

            if got_name == exp_name:
                correct += 1
            else:
                errors.append({
                    "input": conversation[-1]["content"],
                    "expected": expected,
                    "got": result,
                })

            if debug:
                status = "OK" if got_name == exp_name else "FAIL"
                print(f"  [{status}] '{conversation[-1]['content']}' → {got_name} (expected {exp_name})")

    accuracy = correct / total if total > 0 else 0
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="ReachyMini voice orchestrator")
    parser.add_argument("--model", type=str, default="model", help="Model name")
    parser.add_argument("--port", type=int, default=8000, help="SLM server port")
    parser.add_argument("--api-key", type=str, default="EMPTY", help="API key")
    parser.add_argument("--debug", action="store_true", help="Print debug output")
    parser.add_argument("--eval", type=str, default=None, help="Run evaluation on test.jsonl")
    args = parser.parse_args()

    slm = SLMClient(model_name=args.model, api_key=args.api_key, port=args.port)

    # Evaluation mode
    if args.eval:
        print(f"Evaluating on {args.eval}...")
        result = evaluate(slm, args.eval, debug=args.debug)
        print(f"\nAccuracy: {result['accuracy']:.1%} ({result['correct']}/{result['total']})")
        if result["errors"]:
            print(f"\nErrors ({len(result['errors'])}):")
            for err in result["errors"][:10]:
                print(f"  '{err['input']}' → got {err['got']}, expected {err['expected']}")
        return

    # Interactive mode
    orchestrator = ReachyOrchestrator(slm, debug=args.debug)
    print("ReachyMini Voice Assistant (type 'quit' to stop)\n")

    try:
        while True:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit"):
                print("Bot: See you later!")
                break
            response = orchestrator.process_utterance(user_input)
            if response is None:
                print("Bot: See you later!")
                break
            print(f"Bot: {response}")
    except (KeyboardInterrupt, EOFError):
        print("\nBot: See you later!")


if __name__ == "__main__":
    main()
