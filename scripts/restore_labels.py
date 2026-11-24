"""Rebuild runs/fovea/mmerwtrain/labels/*.pt on the server from labels_lite.pt (the fields head ranking reads: id,
views, pixels, scale, tile_letters, correct). The tile labels come from Qwen2.5-VL-3B and are shared by all models."""
from pathlib import Path

import torch

run = Path(__file__).resolve().parents[1] / "runs/fovea/mmerwtrain"
lite = torch.load(run / "labels_lite.pt")
(run / "labels").mkdir(exist_ok=True)
for name, record in lite.items():
    torch.save(record, run / "labels" / name)
print(f"restored {len(lite)} label files")
