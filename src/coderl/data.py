import concurrent.futures
import hashlib
import json
from pathlib import Path
import re
import time
import urllib.request

import pyarrow.parquet as pq
from .schema import Task, TestCase

PAIR_REPO = "stablegradients/pie-gem5-pairs"
PAIR_REV = "8bbdbe1fb21aa0c04bca171421a6c5d8513f6644"
TEST_REPO = "rootacess/pie-perf-testcases"
TEST_REV = "f47671d14853791fcee46bc29a4e2b47fa0ebeb6"


def fetch(url, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return path
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=90) as response, path.with_suffix(path.suffix + ".tmp").open("wb") as out:
                while block := response.read(1024 * 1024):
                    out.write(block)
            path.with_suffix(path.suffix + ".tmp").replace(path)
            return path
        except Exception:
            if attempt == 2:
                raise
            time.sleep(1)


def download_pairs(root):
    root = Path(root)
    for split in ["train", "validation", "test"]:
        fetch(f"https://huggingface.co/datasets/{PAIR_REPO}/resolve/{PAIR_REV}/{split}.parquet?download=true", root / "raw" / f"{split}.parquet")


def public_tests(problem, root):
    directory = Path(root) / "public_tests" / problem
    listing = directory / "files.json"
    fetch(f"https://huggingface.co/api/datasets/{TEST_REPO}/tree/{TEST_REV}/public_test_cases/{problem}", listing)
    files = json.loads(listing.read_text())
    names = {x["path"].split("/")[-1] for x in files if x["type"] == "file"}
    inputs = sorted(x for x in names if re.fullmatch(r"input\.\d+\.txt", x))
    matched = [x for x in inputs if x.replace("input.", "output.") in names]
    if not matched:
        raise ValueError(f"no public test pairs for {problem}")
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        jobs = [executor.submit(fetch,
            f"https://huggingface.co/datasets/{TEST_REPO}/resolve/{TEST_REV}/public_test_cases/{problem}/{name}",
            directory / name) for name in names if name.endswith(".txt")]
        for job in jobs:
            job.result()
    return [TestCase(name=x.removesuffix(".txt"), stdin=(directory/x).read_text(),
                     stdout=(directory/x.replace("input.", "output.")).read_text()) for x in matched]


def prepare(root="data", train=24, validation=6, test=12):
    """Select first unique source per unique problem. Never synthesize missing tests."""
    root = Path(root)
    download_pairs(root)
    counts = {}
    failures = []
    selected_problems = set()
    source_hashes = set()
    pair_counts = {}
    for split, limit in [("train", train), ("validation", validation), ("test", test)]:
        table = pq.read_table(root / "raw" / f"{split}.parquet",
                              columns=["problem_id", "src_id", "tgt_id", "src_code", "tgt_code", "language"])
        pair_counts[split] = table.num_rows
        selected = []
        candidates = []
        locally_seen = set()
        for row in table.to_pylist():
            problem = row["problem_id"]
            digest = hashlib.sha256(row["src_code"].encode()).hexdigest()
            if problem in selected_problems or problem in locally_seen or digest in source_hashes:
                continue
            if not 20 <= len(row["src_code"]) <= 12_000:
                continue
            locally_seen.add(problem)
            candidates.append(row)
        # Fetch modest batches in parallel, retaining stable source order.
        for start in range(0, len(candidates), 8):
            batch = candidates[start:start+8]
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                jobs = [executor.submit(public_tests, row["problem_id"], root) for row in batch]
                for row, job in zip(batch, jobs):
                    try:
                        tests = job.result()
                    except Exception as exc:
                        failures.append({"problem": row["problem_id"], "error": str(exc)})
                        continue
                    if len(selected) >= limit:
                        continue
                    task = Task(task_id=f"{row['problem_id']}:{row['src_id']}", problem_id=row["problem_id"],
                                source=row["src_code"], target=row["tgt_code"], tests=tests, split=split,
                                provenance={"pair_repository": PAIR_REPO, "pair_revision": PAIR_REV,
                                            "test_repository": TEST_REPO, "test_revision": TEST_REV,
                                            "src_id": row["src_id"], "tgt_id": row["tgt_id"],
                                            "source_sha256": hashlib.sha256(row["src_code"].encode()).hexdigest(),
                                            "test_kind": "real CodeNet public input/output"})
                    selected.append(task)
                    selected_problems.add(row["problem_id"])
                    source_hashes.add(task.provenance["source_sha256"])
            if len(selected) >= limit:
                break
        if len(selected) < limit:
            raise RuntimeError(f"only {len(selected)} usable {split} tasks, requested {limit}")
        (root / f"{split}.jsonl").write_text("".join(t.model_dump_json()+"\n" for t in selected))
        counts[split] = {"tasks": len(selected), "public_cases": sum(len(t.tests) for t in selected)}
        print(split, counts[split], flush=True)
    manifest = {"dataset": "PIE C++ / IBM CodeNet", "pair_repository": PAIR_REPO,
                "pair_revision": PAIR_REV, "test_repository": TEST_REPO, "test_revision": TEST_REV,
                "full_pair_counts": pair_counts, "prepared": counts,
                "selection": "first unique problem and source, <=12000 characters, with available paired public tests",
                "split_policy": "prepared subsets exclude overlapping problem ids and source hashes",
                "synthetic_data": False, "download_failures": failures,
                "files": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in root.rglob('*') if p.is_file() and p.name != "manifest.json"}}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def load_tasks(path):
    with Path(path).open() as f:
        return [Task.model_validate_json(line) for line in f if line.strip()]


def load_sft_pairs(path):
    from .schema import SFTPair
    return [SFTPair.model_validate_json(line) for line in Path(path).read_text().splitlines() if line.strip()]
