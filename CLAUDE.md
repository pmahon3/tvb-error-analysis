# tvb-error-analysis: critique of Fousek et al. 2024 (Sci Rep) and the simulations behind it

- **Start here and status:** `doc/LOG.md` (the context file: argument, layout, state of the gates, open items, dated
  entries; it was `README.md` until 2026-11-24). `README.md` is now the short public front page: keep it short, plain
  and written for a human reader; state goes in `doc/LOG.md`. The working
  ledger is `notes/sfm_statistic_indexed_critique.md`; its §6 is the dated gate log.
- **Domain-free core:** `~/Research/Statistics/identification-through-reduction/` (audited). Bring the essay in line
  with it before resuming; `doc/LOG.md` says what is owed.
- **Interventionist remedies** are acknowledged as the remedy and as unavailable here; they are not levelled as a demand.
- **Simulations may be run without asking** (user, 2026-11-18): pre-register, run, score, report. Compute limits unchanged.
- **Layout** (Wilson et al. 2017): `src/` code (flat; `src/README.md` is the index; `src/runall.sh` rebuilds results),
  `data/` inputs, `results/` generated output, `doc/` log, write-up and essay drafts, `external/` third-party copies.
  The log's "Repository" section translates the older paths (`sim/…`, `essay/…`).
- **Run Python with `-I` from inside `src/`** (environment: `src/.venv_vbjax`). The GDPR-restricted dataset (the paper's reference 52) is not used. The
  connectome is the public 76-region TVB one, not the paper's.
- **Git repository since 2026-11-24**, named `tvb-error-analysis` (earlier `critique-fousek2024-sfm`, briefly `sfm-critique`). `notes/` is private and never tracked;
  raw arrays `results/runs/**/*.npz` are not tracked. Commit or push only on request. Before a large edit to an untracked
  file (anything in `notes/`), back it up.
