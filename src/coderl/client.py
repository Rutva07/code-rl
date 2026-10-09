import httpx


class ToolClient:
    def __init__(self, base_url="http://127.0.0.1:8000", transport=None):
        self.http = transport or httpx.Client(base_url=base_url, timeout=180, trust_env=False)

    def close(self):
        self.http.close()

    def _post(self, path, body):
        r = self.http.post(path, json=body)
        r.raise_for_status()
        return r.json()

    def compile(self, source):
        return self._post("/compile", {"source": source})

    def run(self, artifact_id, stdin, timeout=2):
        return self._post("/run", {"artifact_id": artifact_id, "stdin": stdin, "timeout": timeout})

    def compare(self, baseline_id, candidate_id, tests, trials=7, threshold=0.05):
        return self._post("/compare", {"baseline_id": baseline_id, "candidate_id": candidate_id,
                         "tests": [x.model_dump() for x in tests], "trials": trials, "threshold": threshold})
