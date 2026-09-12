"""Portable, checked text inputs for repeatable validation across references."""
import hashlib
import json
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate(payload, identity):
    if payload.get("schema") != 1 or payload.get("identity") != identity:
        raise ValueError("Frozen validation inputs do not match this configuration")
    for scenario in payload["throughput"] + payload["latency"]:
        if "prompt_token_ids" not in scenario:
            if not scenario.get("dataset") or not scenario.get("dataset_split") or scenario.get("output_len", 0) <= 0:
                raise ValueError("Invalid frozen media descriptor")
            continue
        prompts, lengths = scenario["prompt_token_ids"], scenario["output_lens"]
        if not prompts or len(prompts) != len(lengths):
            raise ValueError("Frozen request counts do not match")
        for prompt, length in zip(prompts, lengths):
            if not prompt or not all(type(t) is int and t >= 0 for t in prompt):
                raise ValueError("Invalid frozen token IDs")
            if type(length) is not int or length <= 0:
                raise ValueError("Invalid frozen output length")
            if len(prompt) + length > payload["max_model_len"]:
                raise ValueError("Frozen request exceeds context window")
    return payload


def load_inputs(path: Path, identity):
    envelope = json.loads(path.read_text())
    if digest(envelope["payload"]) != envelope["sha256"]:
        raise ValueError("Frozen input checksum mismatch")
    return validate(envelope["payload"], identity)


def save_inputs(path: Path, payload):
    validate(payload, payload["identity"])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"sha256": digest(payload), "payload": payload}, indent=2))
    tmp.replace(path)
