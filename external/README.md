# Third-party files

Copies of other people's code and public data, kept unmodified so the reference runs can be repeated. Each keeps its
own licence. Nothing here is this project's work.

| Folder | Source | Version | Licence | Used for |
|---|---|---|---|---|
| `pci_fluidity/` | `github.com/ins-amu/PCI_Fluidity` (`utils/`, `snakefiles/`, `hcp-001.zip`) | commit 2bbe4a3, 2025-10-23 | MIT (`LICENSE`) | The INS group's companion pipeline and its 84-region connectome; the TVB reference runs (`src/tvb_reference/tvb_ref_run.py`) call `utils/simulation.py` |
| `tvb_master/` | `github.com/the-virtual-brain/tvb-root` (`nb_mpr.py`, `nb-montbrio.py.mako`) | master as read on 2026-11-24; checksums equal those of the files installed with `tvb-library` 2.10.0 on the run machine | GPL-3.0-or-later (file headers) | The kernel the reference runs executed, kept for reading; not imported by any script |
| `fousek2024/` | Text of Fousek et al. (2024), Sci Rep 14:31970, doi 10.1038/s41598-024-83542-w, extracted from the published PDF | as published | the publisher's; **kept on disk only, not in this repository** | Reading what the paper says (working point, statistics) |
| `tvb_node/` | `github.com/INN-SFU/tvb-node` (notebook text extracted to `.txt`, `sub-0001_weights.txt`) | commit 11f5215, 2026-06-08 | MIT (`LICENSE`) | The lab's workshop notebooks and example connectome, for the zero-delay check |
