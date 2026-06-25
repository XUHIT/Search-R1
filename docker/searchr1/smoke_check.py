import importlib.metadata as metadata

import torch

expected = {
    "vllm": "0.6.3",
    "transformers": "4.47.1",
}

for package, version in expected.items():
    actual = metadata.version(package)
    if actual != version:
        raise SystemExit(f"{package} version mismatch: expected {version}, got {actual}")

for package in ["verl", "flash_attn", "wandb"]:
    __import__(package)

print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("vllm", metadata.version("vllm"))
print("transformers", metadata.version("transformers"))
print("flash_attn", metadata.version("flash-attn"))
print("searchr1 smoke check ok")
