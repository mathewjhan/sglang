"""Experiment: does from_tensors upsert work e2e once the f360b2ac3 wiring is
restored (scope-down commit 8c54787e9 reverted)?

Sequence: base output -> upsert#1 (fresh, Alice adapter) -> upsert#2 (same
name, lora_A zeroed => delta 0, output must return to base if the refresh
really rewrote the pool slot) -> upsert#3 (original weights back).
"""

import json
import os

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

import sglang as sgl
from sglang.srt.managers.io_struct import LoadLoRAAdapterFromTensorsReqInput
from sglang.srt.utils import MultiprocessingSerializer

MODEL_PATH = "Qwen/Qwen3-0.6B"
LORA_REPO = "charent/self_cognition_Alice"
PROMPT = "Hello, my name is"
MAX_NEW_TOKENS = 16


def upsert(engine, name, tensors, config):
    req = LoadLoRAAdapterFromTensorsReqInput(
        lora_name=name,
        config_dict=config,
        serialized_named_tensors=[
            MultiprocessingSerializer.serialize(tensors, output_str=True)
            for _ in range(engine.server_args.tp_size)
        ],
        upsert=True,
    )
    return engine.loop.run_until_complete(
        engine.tokenizer_manager.load_lora_adapter_from_tensors(req, None)
    )


def gen(engine, lora=None):
    kwargs = {"lora_path": [lora]} if lora else {}
    out = engine.generate(
        prompt=[PROMPT],
        sampling_params={"max_new_tokens": MAX_NEW_TOKENS, "temperature": 0.0},
        **kwargs,
    )
    return out[0]["text"]


def main():
    engine = sgl.Engine(
        model_path=MODEL_PATH,
        enable_lora=True,
        max_lora_rank=64,
        lora_target_modules=["all"],
        mem_fraction_static=0.6,
        disable_radix_cache=True,
        log_level="error",
    )
    adapter_dir = snapshot_download(
        repo_id=LORA_REPO,
        allow_patterns=["adapter_model.safetensors", "adapter_config.json"],
    )
    tensors = load_file(os.path.join(adapter_dir, "adapter_model.safetensors"))
    with open(os.path.join(adapter_dir, "adapter_config.json")) as f:
        config = json.load(f)

    base = gen(engine)
    print(f"BASE      : {base!r}")

    r1 = upsert(engine, "alice", tensors, config)
    print(f"UPSERT#1  : success={r1.success} err={r1.error_message!r}")
    assert r1.success, "fresh insert via upsert=True failed"
    alice1 = gen(engine, "alice")
    print(f"ALICE#1   : {alice1!r}")

    zeroed = {
        k: (torch.zeros_like(v) if "lora_A" in k else v) for k, v in tensors.items()
    }
    r2 = upsert(engine, "alice", zeroed, config)
    print(f"UPSERT#2  : success={r2.success} err={r2.error_message!r}")
    assert r2.success, "S1-class failure: second upsert on same name died"
    zero_out = gen(engine, "alice")
    print(f"ZEROED    : {zero_out!r}")

    r3 = upsert(engine, "alice", tensors, config)
    print(f"UPSERT#3  : success={r3.success} err={r3.error_message!r}")
    assert r3.success
    alice2 = gen(engine, "alice")
    print(f"ALICE#2   : {alice2!r}")

    refreshed = zero_out == base
    restored = alice2 == alice1
    distinct = alice1 != base
    print()
    print(f"adapter changed output (sanity)      : {distinct}")
    print(f"upsert#2 rewrote weights in place    : {refreshed}")
    print(f"upsert#3 restored original weights   : {restored}")
    print(f"loaded_adapters after 3 upserts      : {list(r3.loaded_adapters.keys())}")
    verdict = "WORKS" if (distinct and refreshed and restored) else "BROKEN"
    print(f"VERDICT: {verdict}")
    engine.shutdown()


if __name__ == "__main__":
    main()
