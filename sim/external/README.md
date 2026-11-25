# Third-party files

Copies of other people's code and public data, kept unmodified so the reference runs can be repeated. Each keeps its
own licence. Nothing here is this project's work.

| Folder | Source | Version | Licence | Used for |
|---|---|---|---|---|
| `pci_fluidity/` | `github.com/ins-amu/PCI_Fluidity` (`utils/`, `snakefiles/`, `hcp-001.zip`) | commit 2bbe4a3, 2025-10-23 | MIT (`LICENSE`) | The INS group's companion pipeline and its 84-region connectome; the TVB reference runs call `utils/simulation.py` |
| `tvb_master/` | `github.com/the-virtual-brain/tvb-root` (`nb_mpr.py`, `nb-montbrio.py.mako`) | master as read on 2026-11-24 | GPL-3.0-or-later (file headers) | Reading copy of TVB's numba MPR kernel; not imported by any script |
| `tvb_node/` | `github.com/INN-SFU/tvb-node` (notebook text extracted to `.txt`, `sub-0001_weights.txt`) | commit 11f5215, 2026-06-08 | MIT (`LICENSE`) | The lab's workshop notebooks and example connectome, for the zero-delay check |

## Data files outside this folder

| File | What it is |
|---|---|
| `sim/hcp001_conn.npz` | Weights and tract lengths of the 84-region connectome in `pci_fluidity/hcp-001.zip`, as The Virtual Brain configures them through the companion code (weights divided by their maximum). The same two arrays that `sim/tvb_ref_dump.py` saves from the TVB reference set-up into `sim/runs/port/tvb_ref_dump.npz` (checked equal); read by the JAX port. |
| `sim/weights.txt`, `sim/tract_lengths.txt` | The public 76-region connectome distributed with The Virtual Brain (default connectivity). Used by the earlier reimplementation (`sim/mpr_net.py`). The exact upstream file version was not recorded. |
