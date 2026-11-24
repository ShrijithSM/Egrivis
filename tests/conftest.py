import json
from pathlib import Path

import pytest

SCENARIOS = Path(__file__).parent.parent / "scenarios"


@pytest.fixture
def raw():
    """Load a scenario file as a dict the test may edit."""

    def load(name):
        return json.loads((SCENARIOS / f"{name}.json").read_text(encoding="utf-8"))

    return load
