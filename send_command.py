"""Send a single utterance through the Voice SLM → orchestrator → reachy_command."""
import sys
import time

from orchestrator import SLMClient, ReachyOrchestrator

MODEL = "/Users/tsubasa/Documents/TsubasaWorkspace/reachy-voice-slm/fused_model_v2_4bit"
PORT = 8085

slm = SLMClient(model_name=MODEL, port=PORT)
orch = ReachyOrchestrator(slm, debug=True)

text = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("Say: ")
t0 = time.perf_counter()
result = orch.process_utterance(text)
elapsed = (time.perf_counter() - t0) * 1000

with open("/tmp/reachy_command") as f:
    cmd = f.read().strip()

print(f"\n[{elapsed:.0f}ms] → {result}")
print(f"[CMD] {cmd}")
