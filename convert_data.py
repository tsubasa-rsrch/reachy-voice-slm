"""Convert train/test JSONL from question/answer format to mlx-lm chat-with-tools format.

Input format (our format):
    {"question": "[{\"role\": \"user\", \"content\": \"Hello\"}]",
     "answer": "{\"name\": \"greeting\", \"parameters\": {}}"}

Output format (mlx-lm tools format):
    {"messages": [
       {"role": "user", "content": "Hello"},
       {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "greeting", "arguments": {}}}]}
     ],
     "tools": [... tool definitions ...]
    }
"""

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_JOB_DESC_PATH = os.path.join(_HERE, "data", "job_description.json")

with open(_JOB_DESC_PATH) as f:
    JOB_DESC = json.load(f)

TOOLS = JOB_DESC["tools"]


def convert_example(line: str) -> dict:
    """Convert one JSONL line from our format to mlx-lm chat-with-tools format."""
    example = json.loads(line)
    conversation = json.loads(example["question"])
    answer = json.loads(example["answer"])

    # Build messages list
    messages = []
    for msg in conversation:
        if msg["role"] == "user":
            messages.append({"role": "user", "content": msg["content"]})
        elif msg["role"] == "assistant":
            # Previous assistant turn with tool_calls
            tool_calls = msg.get("tool_calls", [])
            assistant_msg = {"role": "assistant", "tool_calls": tool_calls}
            if msg.get("content"):
                assistant_msg["content"] = msg["content"]
            messages.append(assistant_msg)

    # Add the target assistant response (the answer)
    func_name = answer["name"]
    func_args = answer.get("parameters", answer.get("arguments", {}))

    messages.append({
        "role": "assistant",
        "tool_calls": [{
            "type": "function",
            "function": {
                "name": func_name,
                "arguments": func_args,
            },
        }],
    })

    return {"messages": messages, "tools": TOOLS}


def convert_file(input_path: str, output_path: str) -> int:
    """Convert an entire JSONL file. Returns number of examples converted."""
    count = 0
    with open(input_path) as fin, open(output_path, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            converted = convert_example(line)
            fout.write(json.dumps(converted, ensure_ascii=False) + "\n")
            count += 1
    return count


def main():
    data_dir = os.path.join(_HERE, "data")
    mlx_dir = os.path.join(_HERE, "data", "mlx")
    os.makedirs(mlx_dir, exist_ok=True)

    for split in ["train", "test"]:
        input_path = os.path.join(data_dir, f"{split}.jsonl")
        output_path = os.path.join(mlx_dir, f"{split}.jsonl")

        if not os.path.exists(input_path):
            print(f"  Skipping {split}.jsonl (not found)")
            continue

        count = convert_file(input_path, output_path)
        print(f"  {split}: {count} examples → {output_path}")

    # Also create a valid.jsonl (use test as validation for now)
    test_src = os.path.join(mlx_dir, "test.jsonl")
    valid_dst = os.path.join(mlx_dir, "valid.jsonl")
    if os.path.exists(test_src):
        import shutil
        shutil.copy2(test_src, valid_dst)
        print(f"  valid: copied from test → {valid_dst}")

    print("\nDone! Data ready in", mlx_dir)


if __name__ == "__main__":
    main()
