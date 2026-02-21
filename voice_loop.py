"""Voice SLM Loop — continuous listen → understand → respond cycle.

Architecture:
    ReachyMini mic (SSH arecord) → Whisper STT → SLM (intent) → Orchestrator → reachy_command → reachy_hub

Usage:
    python3 voice_loop.py [--duration 3] [--whisper-port 8787] [--slm-port 8085]
"""

from __future__ import annotations

import argparse
import math
import os
import struct
import subprocess
import time
import wave

import requests

from orchestrator import SLMClient, ReachyOrchestrator

ROBOT_HOST = os.getenv("REACHY_HOST", "100.127.25.93")
ROBOT_USER = os.getenv("REACHY_USER", "pollen")
ROBOT_PASS = os.getenv("REACHY_PASS", "root")
WHISPER_URL = "http://localhost:{port}/v1/audio/transcriptions"
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fused_model_4bit")

# SSH with ControlMaster (reuse reachy_hub's persistent connection)
SSH_MUX_DIR = "/tmp/ssh_mux"
SSH_MUX_SOCK = f"{SSH_MUX_DIR}/reachy_%r@%h:%p"
SSH_OPTS = [
    "-o", "StrictHostKeyChecking=no",
    "-o", f"ControlPath={SSH_MUX_SOCK}",
    "-o", "ControlMaster=auto",
    "-o", "ControlPersist=600",
]

# Minimum RMS energy to consider as speech (ReachyMini mic is boosted 10x)
MIN_RMS = 200
# Whisper hallucination filter threshold
MIN_LOGPROB = -0.8


def ssh_cmd(cmd: str, timeout: int = 15) -> subprocess.CompletedProcess | None:
    """Run command on ReachyMini via SSH (reuses persistent connection)."""
    os.makedirs(SSH_MUX_DIR, exist_ok=True)
    try:
        return subprocess.run(
            ["sshpass", "-p", ROBOT_PASS, "ssh"] + SSH_OPTS +
            [f"{ROBOT_USER}@{ROBOT_HOST}", cmd],
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        print(f"  ⚠️  SSH timeout: {cmd[:50]}")
        return None


def scp_download(remote_path: str, local_path: str, timeout: int = 10) -> subprocess.CompletedProcess:
    """Download file from ReachyMini via SCP (reuses persistent connection)."""
    os.makedirs(SSH_MUX_DIR, exist_ok=True)
    return subprocess.run(
        ["sshpass", "-p", ROBOT_PASS, "scp"] + SSH_OPTS +
        [f"{ROBOT_USER}@{ROBOT_HOST}:{remote_path}", local_path],
        capture_output=True, timeout=timeout,
    )


def listen(duration: int = 3) -> str | None:
    """Record audio from ReachyMini mic, return local WAV path or None."""
    remote = "/tmp/reachy_ear.wav"
    ssh_cmd(
        f"arecord -D reachymini_audio_src -f S16_LE -r 16000 -c 2 -d {duration} {remote} 2>&1",
        timeout=duration + 10,
    )

    raw_path = "/tmp/voice_slm_raw.wav"
    output_path = "/tmp/voice_slm_ear.wav"

    dl = scp_download(remote, raw_path)
    if dl.returncode != 0:
        return None

    # Boost gain + highpass (ReachyMini mic is quiet)
    boost = subprocess.run(
        ["ffmpeg", "-i", raw_path,
         "-af", "highpass=f=200,volume=10",
         "-ac", "1", "-ar", "16000", "-y", output_path],
        capture_output=True, timeout=10,
    )
    if boost.returncode != 0:
        subprocess.run(["cp", raw_path, output_path])

    return output_path


def check_energy(wav_path: str) -> bool:
    """Check if audio has enough energy to be speech."""
    try:
        with wave.open(wav_path, "r") as w:
            frames = w.readframes(w.getnframes())
            samples = struct.unpack(f"<{len(frames) // 2}h", frames)
            if not samples:
                return False
            rms = math.sqrt(sum(s * s for s in samples) / len(samples))
        return rms > MIN_RMS
    except Exception:
        return True  # If can't check, assume speech


def transcribe(audio_path: str, whisper_port: int = 8787) -> tuple[str, str]:
    """Send audio to Whisper, return (text, detected_language)."""
    url = WHISPER_URL.format(port=whisper_port)
    try:
        with open(audio_path, "rb") as f:
            resp = requests.post(
                url,
                files={"file": ("audio.wav", f, "audio/wav")},
                data={
                    "response_format": "verbose_json",
                    "temperature": "0.0",
                },
                timeout=30,
            )
        if resp.status_code == 200:
            result = resp.json()
            text = result.get("text", "").strip()
            lang = result.get("language", "")

            # Filter hallucinations by avg_logprob
            segments = result.get("segments", [])
            if segments:
                avg_lp = segments[0].get("avg_logprob", 0.0)
                if avg_lp < MIN_LOGPROB:
                    return "", lang

            return text, lang
    except Exception as e:
        print(f"  [ERROR] Whisper: {e}")
    return "", ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Voice SLM Loop")
    parser.add_argument("--duration", type=int, default=3, help="Listen duration per turn (seconds)")
    parser.add_argument("--whisper-port", type=int, default=8787, help="Whisper server port")
    parser.add_argument("--slm-port", type=int, default=8085, help="SLM server port")
    parser.add_argument("--model", type=str, default=MODEL_PATH, help="SLM model path")
    parser.add_argument("--pause", type=float, default=2.0, help="Pause after response (seconds)")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    # Initialize SLM + Orchestrator
    slm = SLMClient(model_name=args.model, port=args.slm_port)
    orch = ReachyOrchestrator(slm, debug=args.debug)

    print(f"🎤 Voice SLM Loop")
    print(f"   Listen: {args.duration}s per turn")
    print(f"   Whisper: localhost:{args.whisper_port}")
    print(f"   SLM: localhost:{args.slm_port}")
    print(f"   Ctrl+C to stop\n")

    turn = 0
    while True:
        try:
            turn += 1

            # 1. Listen
            print(f"[Turn {turn}] 🎤 Listening...")
            t_start = time.perf_counter()
            wav_path = listen(args.duration)
            if not wav_path:
                print("  ⚠️  No audio captured")
                continue

            # 2. Energy gate
            if not check_energy(wav_path):
                print("  🔇 Silence — skipping")
                continue

            # 3. Transcribe
            text, lang = transcribe(wav_path, args.whisper_port)
            t_stt = time.perf_counter()
            if not text:
                print("  🔇 No speech detected")
                continue

            print(f"  👂 \"{text}\" ({lang}) [{(t_stt - t_start)*1000:.0f}ms]")

            # 4. SLM → Orchestrator → reachy_command
            t0 = time.perf_counter()
            result = orch.process_utterance(text)
            t_slm = time.perf_counter()

            with open("/tmp/reachy_command") as f:
                cmd = f.read().strip()

            print(f"  🤖 {result} [{(t_slm - t0)*1000:.0f}ms]")
            print(f"  📤 {cmd}")
            print(f"  ⏱️  Total: {(t_slm - t_start)*1000:.0f}ms")

            # 5. Wait for reachy_hub to play response
            time.sleep(args.pause)

        except KeyboardInterrupt:
            print("\n🛑 Voice loop stopped.")
            break
        except Exception as e:
            print(f"  ❌ {e}")
            time.sleep(1)


if __name__ == "__main__":
    main()
