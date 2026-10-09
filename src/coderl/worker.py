"""Local compiler worker. Resource limiting is NOT an OS security sandbox."""
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np


def output_equal(actual: str, expected: str) -> bool:
    # Competitive programming whitespace convention; numeric tolerances are not inferred.
    return actual.split() == expected.split()


class Worker:
    def __init__(self, root: str | Path = ".worker", seed: int = 42):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.compiler = shutil.which("g++")
        if not self.compiler:
            raise RuntimeError("g++ is required")
        self.lock = threading.RLock()
        self.seed = seed
        launcher_source = Path(__file__).with_name("launcher.cpp")
        version = hashlib.sha256(launcher_source.read_bytes()).hexdigest()[:16]
        self.launcher = self.root / f"limit-launcher-{version}"
        if not self.launcher.exists():
            subprocess.run([self.compiler, "-O2", "-std=c++17", str(launcher_source),
                            "-o", str(self.launcher)], check=True, timeout=30, capture_output=True)

    def _process(self, argv, stdin="", timeout=2, memory=512, cwd=None):
        # File-backed stdout bounds output memory through RLIMIT_FSIZE.
        with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err, tempfile.TemporaryDirectory(dir=self.root) as tmp:
            metrics = Path(tmp) / "timing.json"
            start = time.perf_counter_ns()
            proc = subprocess.Popen(
                [str(self.launcher), str(timeout), str(memory), str(metrics), *argv],
                stdin=subprocess.PIPE, stdout=out, stderr=err,
                cwd=cwd or self.root, start_new_session=True,
                env={"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"},
            )
            timed_out = False
            try:
                proc.communicate(stdin.encode(), timeout=timeout + 0.2)
            except subprocess.TimeoutExpired:
                timed_out = True
            finally:
                # Also kill descendants after normal parent exit.
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()
            elapsed = (time.perf_counter_ns() - start) / 1e6
            native = json.loads(metrics.read_text()) if metrics.exists() else {}
            out.seek(0)
            err.seek(0)
            return {"returncode": proc.returncode, "timed_out": timed_out,
                    "stdout": out.read(4 * 1024**2).decode(errors="replace"),
                    "stderr": err.read(64 * 1024).decode(errors="replace"),
                    "wall_ms": native.get("wall_ms", elapsed),
                    "host_wall_ms": elapsed, "native_timing": bool(native),
                    "user_cpu_ms": native.get("user_cpu_ms"), "system_cpu_ms": native.get("system_cpu_ms")}

    def compile(self, source: str):
        artifact = hashlib.sha256(("g++ -std=c++17 -O2\n" + source).encode()).hexdigest()
        directory = self.root / artifact
        with self.lock:
            directory.mkdir(exist_ok=True)
            binary = directory / "program"
            if binary.exists():
                return {"ok": True, "artifact_id": artifact, "cached": True, "stderr": ""}
            cpp = directory / "main.cpp"
            cpp.write_text(source)
            result = self._process([self.compiler, "-std=c++17", "-O2", "-pipe",
                                    str(cpp), "-o", str(binary)], timeout=30, memory=1536)
            ok = result["returncode"] == 0 and binary.exists()
            if not ok:
                binary.unlink(missing_ok=True)
            return {"ok": ok, "artifact_id": artifact if ok else None,
                    "cached": False, "stderr": result["stderr"], "wall_ms": result["wall_ms"]}

    def run(self, artifact_id: str, stdin: str, timeout=2):
        # Never resolve client-supplied filesystem paths.
        if len(artifact_id) != 64 or any(c not in "0123456789abcdef" for c in artifact_id):
            raise ValueError("invalid artifact id")
        binary = self.root / artifact_id / "program"
        if not binary.is_file():
            raise FileNotFoundError(artifact_id)
        return self._process([str(binary)], stdin, timeout, cwd=binary.parent)

    def compare(self, baseline_id, candidate_id, tests, trials=7, timeout=2, threshold=0.05):
        rng = random.Random(self.seed)
        samples = []
        with self.lock:
            # Warm up and check both programs on every test before accepting any timing.
            for case in tests:
                for label, artifact in [("baseline", baseline_id), ("candidate", candidate_id)]:
                    run = self.run(artifact, case.stdin, timeout)
                    if run["timed_out"] or run["returncode"] != 0 or not output_equal(run["stdout"], case.stdout):
                        return {"correct": False, "failure": {"program": label, "test": case.name,
                                "actual": run["stdout"][:4000], "expected": case.stdout[:4000],
                                "stderr": run["stderr"], "timed_out": run["timed_out"]},
                                "reward": -1.0, "accepted": False}
            if baseline_id == candidate_id:
                return {"correct": True, "accepted": False, "reward": 0.0,
                        "speedup": 1.0, "runtime_reduction": 0.0, "reduction_ci95": [0.0, 0.0],
                        "samples": [], "identical": True}
            for _ in range(trials):
                row = {"baseline_ms": 0.0, "candidate_ms": 0.0}
                order = [("baseline_ms", baseline_id), ("candidate_ms", candidate_id)]
                rng.shuffle(order)
                for case in tests:
                    for key, artifact in order:
                        run = self.run(artifact, case.stdin, timeout)
                        if run["timed_out"] or run["returncode"] != 0 or not output_equal(run["stdout"], case.stdout):
                            return {"correct": False, "accepted": False, "reward": -1.0,
                                    "failure": {"test": case.name, "program": key, "during_timing": True}}
                        row[key] += run["wall_ms"]
                samples.append(row)
        b = statistics.median(x["baseline_ms"] for x in samples)
        c = statistics.median(x["candidate_ms"] for x in samples)
        ratios = np.array([x["candidate_ms"] / x["baseline_ms"] for x in samples])
        bootstrap = np.random.default_rng(self.seed).choice(ratios, (2000, len(ratios)), replace=True)
        low, high = np.quantile(1 - np.median(bootstrap, axis=1), [0.025, 0.975])
        reduction = 1 - c / b
        # Startup floor dominates tiny programs; do not trust marginal speed differences.
        accepted = bool(reduction >= threshold and low > 0)
        return {"correct": True, "accepted": accepted, "baseline_median_ms": b,
                "candidate_median_ms": c, "speedup": b / c, "runtime_reduction": reduction,
                "reduction_ci95": [float(low), float(high)], "samples": samples,
                "reward": float(np.clip(reduction, -1, 1))}

    def environment(self):
        return {"compiler": subprocess.check_output([self.compiler, "--version"], text=True).splitlines()[0],
                "flags": ["-std=c++17", "-O2", "-pipe"], "python": sys.version,
                "timing": "native CLOCK_MONOTONIC fork/exec/wait4; includes program startup, excludes Python polling",
                "seed": self.seed}
