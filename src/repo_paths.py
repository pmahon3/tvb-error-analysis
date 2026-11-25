"""Where the input files are. In the repository they sit beside src/: inputs in ../data, third-party copies in
../external, per-run output in ../results/runs. On a compute machine the scripts were copied into one flat folder with
the inputs next to them; a file found beside the scripts is used first, so both arrangements work."""
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def _first(*candidates):
    return next((p for p in candidates if os.path.exists(p)), candidates[-1])


def data(name):
    """An input file (connectome)."""
    return _first(os.path.join(HERE, name), os.path.join(HERE, "..", "data", name))


def external(name):
    """A third-party folder, e.g. "pci_fluidity"."""
    return _first(os.path.join(HERE, name), os.path.join(HERE, "external", name), os.path.join(HERE, "..", "external", name))


RUNS = _first(os.path.join(HERE, "runs"), os.path.join(HERE, "..", "results", "runs"))
