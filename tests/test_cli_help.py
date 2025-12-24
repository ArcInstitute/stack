import os
import subprocess
import sys
from typing import Iterable

import pytest


CLI_MODULES: Iterable[str] = (
    "stack.cli.launch_training",
    "stack.cli.launch_finetuning",
    "stack.cli.embedding",
    "stack.cli.generation",
)


def _run_help(module: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [os.path.abspath("src"), env.get("PYTHONPATH", "")]
    )
    return subprocess.run(
        [sys.executable, "-m", module, "--help"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )


@pytest.mark.parametrize("module", CLI_MODULES)
def test_cli_help_exits_cleanly(module: str):
    result = _run_help(module)
    assert result.returncode == 0, (
        f"{module} --help failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "usage:" in result.stdout.lower()

