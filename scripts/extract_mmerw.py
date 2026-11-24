"""Stream MME-RealWorld image archives (tar.gz, possibly split into .part_aa, .part_ab, ...) and extract only the
images used by data_local/raw/mmerw_full/train_split.json into data_local/raw/mmerw_full/imgs/<Image path>.

usage: extract_mmerw.py AutonomousDriving diagram_and_table monitoring_images ocr_cc [remote_sensing]
"""
import json
import os
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "data_local/raw/mmerw_full"


class Chain:
    """Read-only file object over several files in order (a split archive)."""

    def __init__(self, paths):
        self.paths, self.index, self.handle = list(paths), 0, open(paths[0], "rb")

    def read(self, size=-1):
        out = b""
        while size < 0 or len(out) < size:
            chunk = self.handle.read(-1 if size < 0 else size - len(out))
            if chunk:
                out += chunk
                continue
            self.handle.close()
            self.index += 1
            if self.index >= len(self.paths):
                break
            self.handle = open(self.paths[self.index], "rb")
        return out


def main():
    rows = json.loads((ROOT / "train_split.json").read_text(encoding="utf-8"))
    for group in sys.argv[1:]:
        wanted = {os.path.basename(r["Image"]): r["Image"] for r in rows if r["Image"].split("/")[0] == group}
        missing = {b: p for b, p in wanted.items() if not (ROOT / "imgs" / p).exists()}
        parts = sorted(ROOT.glob(f"{group}.tar.gz.part_*")) or [ROOT / f"{group}.tar.gz"]
        print(f"{group}: {len(wanted)} wanted, {len(missing)} missing, archive parts {len(parts)}", flush=True)
        if not missing:
            continue
        with tarfile.open(fileobj=Chain(parts), mode="r|gz") as archive:
            for member in archive:
                name = os.path.basename(member.name)
                if member.isfile() and name in missing:
                    target = ROOT / "imgs" / missing.pop(name)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.extractfile(member) as src, open(target, "wb") as dst:
                        dst.write(src.read())
                    if not missing:
                        break
        print(f"{group}: done, still missing {len(missing)}", flush=True)


if __name__ == "__main__":
    main()
