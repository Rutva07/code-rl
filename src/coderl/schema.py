from pydantic import BaseModel, Field


class TestCase(BaseModel):
    name: str
    stdin: str = Field(max_length=4_000_000)
    stdout: str = Field(max_length=4_000_000)


class Task(BaseModel):
    task_id: str
    problem_id: str
    source: str = Field(min_length=1, max_length=500_000)
    tests: list[TestCase] = Field(min_length=1, max_length=200)
    split: str
    target: str | None = None
    provenance: dict = Field(default_factory=dict)


class SFTPair(BaseModel):
    """Supervised real code pair: compilation tests are not needed for cross-entropy."""
    task_id: str
    problem_id: str
    source: str
    target: str
    split: str
    provenance: dict = Field(default_factory=dict)


class CompileRequest(BaseModel):
    source: str = Field(min_length=1, max_length=500_000)


class RunRequest(BaseModel):
    artifact_id: str = Field(pattern=r"^[a-f0-9]{64}$")
    stdin: str = Field(default="", max_length=4_000_000)
    timeout: float = Field(default=2, gt=0, le=10)


class CompareRequest(BaseModel):
    baseline_id: str = Field(pattern=r"^[a-f0-9]{64}$")
    candidate_id: str = Field(pattern=r"^[a-f0-9]{64}$")
    tests: list[TestCase] = Field(min_length=1, max_length=200)
    trials: int = Field(default=7, ge=3, le=51)
    timeout: float = Field(default=2, gt=0, le=10)
    threshold: float = Field(default=0.05, ge=0, lt=1)
