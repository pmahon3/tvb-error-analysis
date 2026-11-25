"""Where things are in the repository: inputs in data/, third-party copies in external/, per-run output in
results/runs. Paths are absolute, so scripts work from any working directory."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = str(ROOT / "results" / "runs")


def data(name):
    """An input file, e.g. data("hcp001_conn.npz")."""
    return str(ROOT / "data" / name)


def external(name):
    """A third-party folder, e.g. external("pci_fluidity")."""
    return str(ROOT / "external" / name)
