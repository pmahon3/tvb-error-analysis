# Working log

The author's own running record: context, pre-registered checks and dated results, oldest first. It was `README.md`
until 2026-11-24, so earlier entries and outside pointers that say "README" mean this file. It is not written for
readers; start with `README.md`.

## Critique of Fousek et al. 2024, "Symmetry breaking organizes the brain's resting state manifold" (Sci Rep)

Context file for this folder. Compass area: `inn-science` (`~/Compass/areas/inn-science/STATUS.md`).
Started 2026-11-16 inside the maths repo (`notes/literature_review/lab_critiques/`), moved here 2026-11-17.
Lab meeting on this paper: **Wednesday 2026-11-18, 9:30–10:30 PT** (agenda in `~/Downloads/Agenda for lab meeting Nov 18, 2026.docx`:
comprehension questions, then interpretation: frequency vs in-strength (Fig 5D), "how can we apply SFM in our work", modelling limitations).

## Reversal, confirmed 2026-11-24: the "transition regime" at (G, σ) = (1.05, 0.30) belongs to the dt = 0.05 scheme
Every simulation at that point used an integration step of 0.05 with the rate projected to r ≥ 0. That scheme
switches there (frac_up 0.30). At dt = 0.025 and 0.0125 the same parameters sit on the upper plateau (frac_up 0.56,
0.59) with brief dips. Seen through both simulators (JAX and `mpr_net.py`) at all three steps; hostile check in
`notes/OPERATING_POINT_DT_REVERSAL_CHECK.md`. Consequences:
- Claims that the regime EXISTS at or near that point (G0e, band and boundary claims, the rewiring contrast, ridge
  check v1's band, Figure 2 draft panel B) are claims about the dt = 0.05 scheme, withdrawn for the equation.
- Claims about statistics computed ON those records (G1 against both nulls; the two-state rival `g3_trans`) keep their
  logic; the generator is "the dt = 0.05 scheme, conditioned on staying finite".
- The description "6–8 transitions, high 10–13 s, low about two minutes" was a detector artefact at dt = 0.05 itself.
  Re-scored (amended detector; verified on `runs/g0e_alt_s{1,2}.npz`): 22 and 22 transitions, up ≈ 26 s, down ≈ 29 s,
  frac_up 0.29 and 0.34. The two-state rival then has 1 transition against 22, not "2 vs 6–8".
- Unaffected: stage 0 (hysteresis) and the structural probe (noise-free).
- NOT yet updated (the user's text): `essay/v3/essay_v3.tex` (lines listed in the check, §3), its Figure 3
  (`sim/fig_f2.py`), `notes/figures/fig2_tvb_draft` panel B, `fig2_tvb_generic` Table 1.
- Untested: whether the other gates' regime labels (all run at dt = 0.05) move with the step.
Records: the dated entries of 2026-11-24 below; the ledger's §6 carries the same note. Backups of both files before
this propagation: `*.bak_before_dt_reversal_2026-11-24`.

## What this is
A constructive critique for an INN lab meeting, written as a short essay in the user's voice. It is not a take-down.
The method is charitable reconstruction first (state the authors' position better than they did), then one objection,
then the positions that remain. That method is a working rule and is **not named in the essay**.

The argument, in one paragraph. The paper fits a whole-brain network model (Montbrió–Pazó–Roxin nodes on a
connectome) so that its simulated BOLD reproduces resting-state statistics (switching index of windowed dFC, edge
events, low-dimensional structure), then reads the fitted model as a mechanistic explanation. In Craver's terms the
paper establishes a how-possibly explanation and presents it as how-actually. In Corless–Fillion's terms, a small
backward error (the data are the exact output of a nearby model) is reported where a small forward error (the
mechanism is close to the brain's) is needed, and the gap between them is the condition number of the statistics
used, which here is unbounded because those statistics have a large kernel (stationary processes with the same
second moments reproduce them). The two readings line up almost term for term; that pairing is the one thing
not found in print. Interventionist remedies are acknowledged as unavailable here; the essay lists what remains.

## Layout
- `essay/` — `essay.tex` (pdflatex + biber; 15 pp), `refs.bib` (43 entries, verified), `essay.pdf`, `fig_g1.pdf`
  (made by `sim/fig_g1.py`; the only data figure in the body; all floor numbers in the essay come from its ensembles). Build:
  `pdflatex essay && biber essay && pdflatex essay && pdflatex essay`. `.gitignore` keeps only sources and the PDF.
  Two `.tex.bak` files are earlier versions (v2; v3 before the sentence-length pass).
  `essay/v3/` (2026-11-18) continues the v2 line: all `[PRO]` steps written out (two as appendix lemmas, `lem:phase`
  and `lem:floor`); `[NUM]` and `[HEDGE]` markers remain until gate G0e and the remaining numbers are in.
  **Audit passes 1–3 run the same day** (`essay/v3/audit/`: PROOF_CHECK, PAPER_CLAIMS_CHECK, REFS_CHECK, WITNESSES_A/B,
  collated in CORRECTIONS.md) and applied. Three reversals: (b)Θ₃ is approximate, not exact (overlapping windows: no
  record map reorders them; the witness is the surrogate's small KS distance); the "floor" lemma explains the null's
  size and does not bound it, so s₀ is the surrogate value; the Θ₂ witnesses are Messé/Nozari for the covariance
  statistic and Deco 2017 + Liégeois 2017 for the histogram, with Hansen 2015 against. Still open: Laumann 2017
  unread (needs SFU proxy), Deco 2017 figure values.
  **2026-11-19:** gates G0e/G0f scored (a first G0f verdict by raw crossings was wrong and is recorded as withdrawn);
  the paper's kind of regime is at (G, σ) = (1.05, 0.30) **[2026-11-24: of the dt = 0.05 scheme only; see the
  reversal section at the top]**; G1 there under phase and AR(1) nulls: fluidity index and
  histogram both separate the model from its null (×2–4; KS ≥ 0.15). v3 updated: Figure 3 (F2, two regimes), Table 1
  dagger note, Appendix C gate table, hedges resolved except (c)Θ₂ (rivals). 8 NUM slots remain (see ledger).
  **Later 2026-11-19:** all simulations complete; v3 has no markers left. Rewiring at the transition regime removes the
  regime (point for the authors, model-internal) **[2026-11-24: within the dt = 0.05 scheme; untested at finer steps]**; the two-state reduction with fitted hazards attains the fluidity index
  and the explained-variance share from one start and is separated by the histogram (KS 0.36–0.42). (c)Θ₂ resolved
  accordingly. **New line: `essay/v4_theory/`** — a theory-level skeleton aimed at SFM/synergetics itself (universality vs
  mechanism; Jirsa 2020 quotations; Kaplan–Craver vs Batterman–Rice/Ross; the identification face of the slow–fast
  split). Reads and scout in `notes/theory/`. Quotations from Batterman–Rice and Kaplan–Craver are via secondary sources
  until the originals are read (marked [CHECK]); the Ross–Jirsa–McIntosh 2025 citation is incomplete.
  **2026-11-19 later: SPLIT.** The domain-free core of v4 (§3–5) now lives in `~/Research/Statistics/identification-through-reduction/`
  (`paper/core.tex`, README there). This folder keeps the domain application (v4 §2, Table 1, the three senses of
  `mechanistic', the VEP material in the backups) and is PAUSED until the core has been prior-art-audited.
  Proof-check fixes of 2026-11-19 (Prop 1 for the whole fibre; Prop 2/3 constant-rank; leak two-part; §4.4 O(T) at γ⁻⁴)
  are applied in both copies.
  **2026-11-19 (after the revert): v4 reformed on the restored base.** Opening ¶3 rewritten; new §4 "The critique
  relative to a target": four-parameter small case (eq:small), identified algebra 𝒜_ρ + Proposition 2 (g identified iff
  g∈𝒜_ρ, ⟦HAND⟧), admissible set + family of reductions + Proposition 3 rank condition (⟦HAND⟧), finite-sample
  resolution with the recurrence qualification (stationary records carry O(1) information on transient-only leak
  directions), class claims as tests, the fibre as the framework's empirical content, and Table 1 classifying the
  authors' claims (Jirsa 2020, Fousek 2024, VEP). New §5 skeleton O1–O6 of what is missing (O6 = information-geometry
  intuition: Fisher metric degenerate on ker Dρ; gap = Fisher metric vs an intervention metric), Limits, Conclusion.
  The explanatory-status / replies / two-positions sections are CUT (in the backups). Next: decide per O-item whether a
  generic-case solution exists; explanatory status deferred until then.
  **2026-11-18 22:35 MST: REVERTED by the user.** `essay_v4_theory.tex` was restored to the 15:09 snapshot
  (`essay_v4_theory_before_restructure.tex.bak`: skeleton + mathematical explication + generic figures, before the
  sectioning restructure). Everything described in the next paragraph lives only in
  `essay_v4_theory_2026-11-18_2230_reverted.tex.bak` and the intermediate `.tex.bak` files; the five epilepsy bib
  entries (verified DOIs) remain in `refs.bib`, uncited. Record of what was tried and withdrawn:
  v4 restructured several times with the user: argument-first sectioning; proof of the
  Proposition written out as prose; voice pass from the Guelph samples; each perspective (Haken, Jirsa, Craver,
  Batterman–Rice/Ross) explicated in the user's fixed form (entry by need, one worked quotation, "in short",
  enumerated claims, the small case, "so"), the two-variable "small case" carried through every section; then ported
  to article register (abstract; Introduction/Background/Non-identification/Explanatory status/Objections/Discussion/
  Conclusion; labelled premises S1–S3, F1–F3, M1–M3, D1–D3, U1–U3, I1–I3, E1–E3; Definition and Example environments).
  **Added:** three senses of "mechanistic" (interventionist/generative/realist) and "non-reductive mechanistic"
  defined in §2.2; the Virtual Epileptic Patient / EPINOV as §2.4 (the framework's instance that meets the
  interventionist sense; trial outcome NOT verified); §4.3 "Nesting"; Objection 4. Five new bib entries (jirsa2014,
  jirsa2017, wang2023, saggio2020, epinov) are marked unverified; a sonnet scout was launched to verify them.
  The port mapping (essay habit → article device) is in `~/Research/Philosophy/essay_sources/writing_samples/style_notes.md` §12–13.
  `essay/skeleton_v2/` is a **separate line**: the pre-results v2 skeleton (XeLaTeX, Georgia) with the mathematical
  argument worked in (Eqs. 1–4, Proposition, Appendix A lemma), no results, no figures, seven `[NUM]` slots, cut 20 %
  by whole-sentence deletion on 2026-11-17 (2718 prose words; `essay_skeleton_v2_before_cut.tex.bak` is the uncut
  text). The interventionist "third reason" was dropped in the cut, per the standing rule.
- `notes/` — `sfm_statistic_indexed_critique.md` is the **working ledger**: ground (statistic-indexed argument,
  Ψ/φ/π, backward error β, condition number), invariance lemmas, pre-registered gates G0–G5, and §6 the dated gate log
  with results. `fousek2024_sfm_critique.md` is the reading notes on the paper. `sfm_craver_essay_skeleton.md` (+pdf)
  is the superseded v1 outline. `email_fillion_draft.md` is a draft email to Prof. Fillion (SFU) for the user to edit.
- `sim/` — reimplementation used for the numerical gates: `mpr_net.py` (MPR network + Balloon–Windkessel; connectome
  normalised to unit mean in-strength; within-row permutation), `stats.py` (the paper's statistics from its Methods),
  `run_sweep.py`, `g1_surrogate.py` (cross-spectrum-preserving phase randomisation), `ising_net.py` + `fit_rates.py` +
  `escape_times.py` (two-state reduction, G3), `rebold.py` (offline BOLD). `weights.txt`/`tract_lengths.txt` = the
  76-region public TVB connectome (not the paper's HCP one). `runs/` ≈ 550 MB of `.npz` outputs and logs.

## Repository (since 2026-11-24)
The folder was `critique-fousek2024-sfm`, then `sfm-critique`, and is `tvb-error-analysis` since the evening of 2026-11-24
(links at both old names keep earlier pointers working). It was put under git that day and laid out after Wilson et al.
2017 and Noble 2009 (`src`, `data`, `results`, `doc`; scout report in the session of 2026-11-24). **Paths in the dated
entries of this file are as written at the time.** Translation:

| Path in older entries | Now |
|---|---|
| `README.md` (this log) | `doc/LOG.md`; `README.md` is the short public front page |
| `sim/*.py` | `src/{core,sweep,validation,tvb_reference,earlier_studies/…}/*.py`; eight scripts renamed (table in `src/README.md`) |
| `sim/runs/Aprime/` | `results/runs/tvb_reference/` |
| `sim/runs/port/` | `results/runs/validation/` |
| `sim/runs/sweep/` | `results/runs/sweep/` |
| `sim/runs/` (files at the top level), `sim/runs/ridge*/` | `results/runs/earlier_studies/{first_reimplementation,two_state_reduction,ridge_checks}/`; run logs in a `logs/` folder beside the records |
| `sim/external/…` | `external/…` |
| `sim/weights.txt`, `sim/tract_lengths.txt`, `sim/hcp001_conn.npz` | `data/…` |
| `essay/…` | `doc/essay/…` |
| `notes/figures/sweep_2026-11-24/`, `figures/…` | `results/figures/{report,maps,derivatives}/`, `results/tables/` |
| `notes/SWEEP_WRITEUP_2026-11-24.html`, `reports/…` | `doc/sweep_writeup.pdf` |

- **Tracked:** everything except the items below. `src/runall.sh` rebuilds figures, tables and the write-up from
  `results/runs`. The code was then grouped by study and formatted with black (commits `1b64d7e`, `186cd11`); after each
  step: imports resolve, `runall.sh` gives identical figures, `mprjax_match.py` matches TVB (2.3e-12), and a fixed small
  job through `mprjax_run.py` gives identical statistics. As-run provenance (file checksums against the tower, container
  digest, launch commands) is in `src/README.md`.
- **Not tracked** (`.gitignore`): `notes/` (working notes, lab-meeting notes, drafts of correspondence), the local
  Python environment (`src/.venv_vbjax`), backups `*.bak*`, and the raw arrays `results/runs/**/*.npz` (about 1.5 GB)
  except `runs/validation/tvb_ref_dump.npz` and `runs/validation/traces_wp.npz`.
- Entries that point into `notes/` (the ledger, the reversal check, the reading notes) refer to files that exist only
  on this machine.
- Licences (user, 2026-11-24): MIT for the code (`LICENSE`), CC BY 4.0 for text, figures and results (`LICENSE-TEXT.md`);
  third-party files keep their own.

## State of the gates (see ledger §6 for numbers)
- G0 positive control: pass (bistable regime found; three collective basins: all-down, mixed, mostly-up).
- G1 stationary surrogate: fires in the mixed state (the statistics do not separate the model from its own
  stationary Gaussian surrogate), attained in the mostly-up state.
- G2 within-row rewiring: fires in the authors' favour (in-strength alone does not fix the basins); confound: the
  permutation lowers λ₁(W) by 12%. λ₁-matched follow-up (G = 1.03, permutations 11 and 12, three starts each):
  partial return, mixed state 3/6, all-down 3/6, mostly-up 0/6 (original: up-containing 6/6). In-strength + λ₁ do
  not reproduce the basins. Done 2026-11-17 late evening; essay and ledger updated.
- G3 two-state reduction: partial (captures bistability and mostly-up, not the basin boundary or the mixed state).
- Stationary floor lemma (Appendix A): proved; the plug-in bound has an honest ×2–4 gap.

## Rules carried over from the session
- Keep the user's voice; reorder, do not rewrite, unless asked. Short sentences (median ≈ 22 words).
- No metadiscourse ("in this section we…"). Captions describe what the figure shows; they do not interpret.
- Formal elements: Observations in prose or tables; one real Lemma (appendix); Proposition in Rothenberg form.
- Interventionist methods are acknowledged as the remedy and as unavailable; they are not levelled as a demand.
- Pre-register every numerical gate before running it; record results dated in ledger §6; never backfill.
- Compute: ≤ ~2 GB, one heavy job at a time, always under a timeout (`perl -e 'alarm shift; exec @ARGV' SECS …`).
- Run python with `-I` from inside `sim/`. The GDPR-restricted dataset (paper ref 52) is not used.
- Nothing here goes to anyone outside until the user has rewritten it in their own words.

## Open items
0. **[2026-11-25: no longer applies. The essay is parked in `archive/essay/`; see the entry of that date.]**
   **2026-11-22 (reviewed: false).** The domain-free core was audited (2026-11-19) and repaired (2026-11-21). Before
   resuming, bring essay v3 in line with it: scope the "unbounded condition number" wording (unbounded only along
   directions with no leak; finite and growing with the time-scale separation along leaky ones), and cite the owners the
   audit named (Rothenberg; Transtrum–Qiu). Which kind the paper's mechanistic claims fall under is not established.
1. Email to Fillion: user fills [title]/[email], confirms the Corless wording; optional SFM-ontology paragraph
   (proposed in the session, not inserted).
3. Possible lab-meeting handout: the three-item list in essay §4 and Fig. 1–3 stand alone.

## 2026-11-23, lesion check on Pillai & Jirsa 2017 Fig. 4 (user: "press on 1"; registered before running)
Setting (paper p.1018–1019): 3-D networks q' = R(Pq)q + μ w F(Pq) with P w = I₂ reproduce the same 2-D Excitator flow
ξ' = Rξ + μF(ξ), ξ = Pq. The fibre for a record of ξ is {(P, w): P w = I}. Paper's pair: Pa, Pb share w = [e1 e2] and
differ in P.
⟦HAND⟧ derivation: lesioning node k gives ξ' = Rξ + μ(I − M_k)F(ξ), M_k = p_k w_kᵀ (column k of P, row k of w), on the
range of P without column k. Σ_k M_k = I is all the reduced flow fixes; each M_k varies along the fibre.
Predictions: (1) the identity holds through the network equations to rounding (prior 0.97; failure = misread equations);
(2) for the paper's own pair, M_k is the SAME for every k (they share w and columns 1–2 of P), so the lesion effect on
the reduced field is identical; only the lesioned state space differs (a: q1 or q2 lesion collapses it to a line;
b: stays a plane) (prior 0.9); (3) a third network c on the same fibre (P as b, w with w₃ ≠ 0) makes node 3 disruptive
where it is harmless in a and b (prior 0.95).
Scope: the fibre is relative to a record of the functional variables ξ only. A record of the node activities q
shrinks it (the paper's own Fig. 4D separates a and b by their transients in q).
Short hand argument already explains (2)–(3); the run is a check of the reading, not evidence for a general case.
Script: `sim/pillai_jirsa_lesion.py`.
**Outcome (2026-11-23, same day).** (1) HELD: identity error ≤ 4e-16 in every case (a check of the reading, since the
network route and the formula are the same algebra). (2) HELD: Pa and Pb have identical M_k (diag(1,0), diag(0,1), 0);
the lesioned FIELDS agree, but lesioning q1 collapses a to a line while b keeps a plane with mean deviation angle 0.42
(q2: 0.08). So the lesion's behavioural effect already differs between the paper's own pair, through the lesioned
readout's range. (3) HELD: in c, node 3 is the most disruptive (deviation angle 0.23; 0 in a and b); q1 0.06, q2 0.15.
Deviation angle here = mean angle/π between lesioned and intact fields on |ξ| = 1; not the paper's exact DA (not given).
Reading (⟦HAND⟧): interventions on the shared law (a, I, b, τ, μ) act through the reduced flow and are fibre-constant;
node-addressed interventions (lesion, stimulation of node k, which adds p_k·u) act through p_k and w_k and are not.
"Functionally disruptive lesions can be identified" therefore needs the fibre position, not the flow.
Not tested: whether the paper's Fig. 5 averages (random lesion sets, large N) are fibre-insensitive in distribution.

## 2026-11-23, Fig. 5 averages: is lesion robustness fibre-constant? (user: "proceed"; registered before running)
Paper: networks of 50–10,000 nodes, embedding modes drawn uniform(0,1) then orthogonalised; 5–25% of nodes removed, 100
lesion sets each; caption 5B: "As the network size doubles, the robustness increases almost exponentially."
Model here: orthonormal embedding w (N×2), readout P = wᵀ, so M_S = Σ_{k∈S} w_k w_kᵀ. On the manifold |ξ| = 1, R = 0 and
the lesioned field is μ(I − M_S)F. DA = mean angle/π between (I − M_S)F and F over |ξ| = 1.
⟦HAND⟧: E_S[M_S] = f·I for a uniformly random set of fraction f, so the MEAN lesioned field is μ(1 − f)F, parallel to
the intact one (forced; not evidence). DA comes from the fluctuation M_S − fI, whose size is set by how the function is
spread over nodes: participation m ≈ 1/Σ_k ‖w_k‖⁴ (inverse participation ratio of the embedding), not by N itself.
Predictions: (P1) delocalised embeddings (paper's construction): DA ∝ N^(−1/2) at fixed f, i.e. ratio per doubling
≈ 0.71 (log–log slope −0.5 ± 0.1), which would read as "almost exponentially" per doubling. Prior 0.8.
(P2) embeddings localised on m nodes (same flow, same fibre): DA flat in N at fixed m (slope 0 ± 0.1), prior 0.9;
and DA ∝ m^(−1/2) at fixed N (slope −0.5 ± 0.15), prior 0.75.
Defeat: P1 slope outside [−0.6, −0.4], or P2 DA falling with N at fixed m.
Script: `sim/pillai_jirsa_lesion_fig5.py`. Sizes up to N = 3200; memory trivial (N×2 matrices).
**Outcome (2026-11-23).** All HELD. (P1) delocalised: log–log slope −0.495, −0.516, −0.506 at f = 0.05, 0.15, 0.25; ratio
per doubling 0.70–0.71. (P2a) localised on m = 10 nodes: DA flat in N (slopes −0.012, +0.011, −0.009; no collapsed
lesion sets). (P2b) N = 1600: DA ∝ m^(−0.45…−0.51). Consistency: m = N = 1600 reproduces P1 at N = 1600.
Hand explanation (which withdraws the run as evidence for anything general): I − M_S = (1 − f)I − (M_S − fI), and the
fluctuation has size ≈ √(f(1 − f)/m_eff), so DA ≈ c·√(f/((1 − f)·m_eff)); the predicted f-ratio 0.25 vs 0.05 is 2.52,
observed 2.56 at N = 3200.
Reading: the paper's "robustness increases almost exponentially as N doubles" is √2 per doubling, and it comes from the
participation m_eff of the embedding, a fibre coordinate, not from N or from the flow. Same flow, same N: spread
embedding robust, embedding on 10 nodes fragile at any N. Their construction (dense random modes) chooses the robust part
of the fibre. Relocated: node-level records fix w (activity amplitudes on the manifold) but leave P free subject to
Pw = I; M_k = p_k w_kᵀ needs both. Smallest open case: what a node-level record fixes about P. Not started.

## 2026-11-23, what a node-level record fixes about the readout P (user: "continue with 1"; registered before running)
⟦HAND⟧ Write P = w⁺ + Y U⊥ᵀ, where w⁺ is the pseudo-inverse of w, U⊥ an orthonormal basis of range(w)^⊥ and Y a free
2×(N−2) matrix; then P w = I for every Y. Split q = wξ + h with ξ = Pq and h ∈ ker P; the network gives
ḣ = R(ξ) h exactly. On the limit cycle ⟨R⟩ = −μ⟨ξ·F/|ξ|²⟩ ≈ −1.1μ here (τ = 3, b = 0.1), so h relaxes slowly.
- Noiseless record on the attractor: q(t) = wξ(t). It fixes w (given the reduced law's coordinates) and P only through
  P w = I: Y is invisible. Pillai–Jirsa's Pa, Pb are this case: same w, Y = 0 vs Y ≠ 0, identical attractor at the node
  level, different transients (their Fig. 4D).
- Lesion effect: p_k = w⁺e_k + Y U⊥ᵀe_k, so M_k = p_k w_kᵀ depends on Y unless node k lies in range(w).
- Noisy record (additive σ on each node): h makes excursions of variance ~σ²/(2·1.1μ); the drift's sensitivity to Y is
  proportional to U⊥ᵀq, i.e. to those excursions. Girsanov information per unit time ~ (σ²/μ)/σ², independent of σ.
  So noise probes the fibre: Y is identified at the regular rate 1/√T for any σ > 0, and not at all at σ = 0.
Predictions (N = 4, μ = 0.1, random orthonormal w, true Y with entries O(1)):
(Q1) noiseless attractor trajectories of (w, Y = 0) and (w, Y_true) agree at the node level after the transient
(max difference < 1e-6 of the amplitude), prior 0.9; (Q2) single-node lesion damage differs between them for every
node not in range(w), prior 0.95; (Q3) with noise, sd(Ŷ) ∝ T^(−1/2) (slope −0.5 ± 0.15), prior 0.85; (Q4) sd(Ŷ)·√T
roughly constant across σ ∈ {0.02, 0.05, 0.1} (within a factor 1.5), prior 0.6. Defeat of "noise probes the fibre":
sd(Ŷ) growing like 1/σ as σ falls.
Script: `sim/pillai_jirsa_readout.py`. Euler–Maruyama, dt = 0.01; drift-regression MLE (least squares on increments).
**Outcome (2026-11-23).** (Q1) HELD: node-level difference 2.1e-14 of the amplitude. (Q2) HELD: lesion DA differs for all
four nodes (Y = 0 vs Y_true: 0.135/0.115, 0.044/0.109, 0.048/0.078, 0.133/0.145). (Q3) FAILED as registered: sd(Ŷ) fell
faster than T^(−1/2) from T = 250 to 1000 (log–log slopes −0.79, −1.01, −0.66 at σ = 0.02, 0.05, 0.1); only 8 replicates,
so each slope carries roughly ±0.25. (Q4) FAILED: sd·√T at T = 1000 is 1.41, 0.60, 2.13 for σ = 0.02, 0.05, 0.1, a
factor 3.5, not monotone in σ. The defeat condition for "noise probes the fibre" (sd growing like 1/σ as σ falls) is NOT
met: σ = 0.02 does better than σ = 0.1. So: Y is identified from noisy node-level records at every σ tried, but my
σ-independence estimate is wrong in detail; the non-monotone σ dependence is UNEXPLAINED (candidates, unchecked:
Euler/drift-regression bias at small σ with dt = 0.01; nonlinearity of R at large excursions; small-sample nonlinearity
at T = 250). Owner noted: the paper's own footnote already quotes Transtrum et al. 2016, "Phenomenological models have
limited predictive power for experiments that manipulate microscopic control knobs", which is the lesion result in words.

## 2026-11-23, ridge check: do the fitted statistics see G and η separately? (user: "proceed with the check"; registered before running)
Model: `sim/mpr_net.py` (MPR network from the paper's Methods; η = −5, J = 15, Δ = 1, τ = 1; 76-region TVB connectome,
unit mean in-strength; Balloon BOLD). Operating point: the one paper-like cell in hand, (G, σ) = (1.05, 0.30) (low state
with brief global high-activity intervals about every 2 min; `notes/sfm_statistic_indexed_critique.md`, G0e).
**[2026-11-24: that description is a detector artefact, and the point switches only in the dt = 0.05 scheme; see the
reversal section at the top.]**
Question: near that point, do the statistics the paper fits respond to G and to the node excitability η separately, or
only through one combination (a ridge = a fibre of the statistic map in the (G, η) slice)?
⟦HAND⟧ prior reasoning: η and G·(input) enter the v-equation additively. In the low state the input is G·r_dn·s_i
(r_dn ≈ 0.08), in the high state G·r_up·s_i (r_up ≈ 1.03): the two states weigh G against η differently, and G also
shapes the spatial pattern (in-strength s_i) while η is uniform. So a strict ridge is not expected; a sloppy direction is.
Design: 5 configurations (centre; G ± 0.03; η ± 0.1) × 6 seeds, common random numbers within a seed (same initial state
and noise for all 5), 600 s model time, dt 0.1 ms. Statistics (k = 6): switching index, VAF2, mean static FC, corr(FC, SC)
over the upper triangle, mean up-fraction, number of debounced transitions (up-fraction smoothed over 2 s, flips on
crossing 0.6 up / 0.4 down; node up if r > 0.5). Central differences per seed; each statistic scaled by its across-seed
sd at the centre (what one record resolves); j_G, j_η = whitened sensitivity vectors; c = |cos(j_G, j_η)|, bootstrap
over seeds (2000 resamples).
Outcomes: RIDGE if c > 0.95 and bootstrap 5th percentile > 0.9; SEPARATE if c < 0.8 and 95th percentile < 0.9; else
INCONCLUSIVE. Priors: RIDGE 0.35, SEPARATE 0.35, INCONCLUSIVE 0.30 (rare-event noise). Validity check: if any ±
configuration loses the regime (no transitions, or all-up) in most seeds, the derivative is a regime jump: report, do
not score. Scale-invariant by construction (cosine of whitened columns). Not tested: σ (the paper's second tuning
coordinate), other node parameters (J, Δ).
Script: `sim/ridge_check.py`. Memory ~0.1 GB (r kept at 100 ms). Wall time estimated from a 20 s smoke run first.
**Amendment before any scoring (2026-11-23).** dt = 0.1 ms is numerically unstable at η = −4.9 (overflow within ~2 s in
3 of 3 seeds; `sim/ridge_diag.py`, tower runs `diag_dt*.txt`); at dt = 0.05 and 0.025 all 15 configuration-seeds stay
finite, and the peak rate grows as dt falls (centre, seed 1: 2.13 → 2.54 → 3.02), so dt = 0.1 also under-resolves the
excursions. The check runs at dt = 0.05 for all configurations. Seeds run as six parallel processes on the tower
(common random numbers are within a seed, so nothing changes). Everything else as registered. Caveat recorded: the
operating point (1.05, 0.30) was characterised in earlier runs; the validity check will show whether it still has
transitions at dt = 0.05.

## 2026-11-23, TVB instantiation of the domain-free note (`tvb_port/`)
`tvb_port/core_tvb.tex` is a copy of `~/Research/Statistics/identification-through-reduction/paper/core.tex` with a TVB
dictionary added (subtitle; one Opening sentence; a §1 table: term → TVB object; four sentences). `core_tvb.diff` is the
diff against the source. Figures and `refs.bib` are read from the source folder by relative path (pointer, not copy);
`refs_tvb.bib` adds Sanz Leon et al. 2013 and Schirner et al. 2018. 25 pp, builds clean. No result changed. If the
source note changes, regenerate the copy from it and reapply the diff.
**Deleted 2026-11-23, same evening, at the user's request** ("lets just keep working with the original"). `tvb_port/` no longer exists; the source note is the only copy.
- 2026-11-23, draft figure: `notes/figures/fig2_tvb_draft.{tex,pdf,png}` — the note's Figure 2 drawn for the TVB case (A: (G, η) slice with fibres DASHED, pending the ridge check; B: two-state switching description at (1.05, 0.30); C: three laws of the BOLD record (MPR network, two-state model, phase surrogate); D: grid of which statistic separates which, from gates G1/G3). Not for sharing.
- 2026-11-23: generic version `notes/figures/fig2_tvb_generic.{tex,pdf,png}` (structure only: tuning slice, reduced description, laws of the BOLD record including other model classes, fitted statistic) with Table 1 of instances: Fousek 2024 (runs), Schirner 2018 (body: diagonal trade-off and 'single optimal subspace' verified in the PDF; the scout's 'We confirmed identifiability' quote NOT found in the main text), Hashemi 2024 (scout), Pillai–Jirsa 2017 (body; runs). Not for sharing.
**Outcome (2026-11-23, late): NOT SCORED, by the registered validity rule.** Tower, 6 seeds × 5 configurations, 600 s,
dt = 0.05 (`runs/ridge/ridge_s*.json`).
- Regime lost: G − 0.03 has 0 transitions in 6/6 seeds (frac_up 0.09), η − 0.1 has 0 in 6/6 (frac_up 0.03); η + 0.1 has
  exactly 2 in 6/6; G + 0.03 has 2–6. Centre: 4–18 transitions, frac_up 0.31–0.34, VAF2 0.977–0.979 (matches G0e).
  The operating point sits in a band narrower than ±3% in G and ±2% in η; one-sided moves are regime jumps, so the
  registered central differences are not derivatives.
- Numerics: BOLD is non-finite in 8 of 30 configuration-seeds (centre 1/6, G+ 2/6, η+ 5/6; seen as VAF2 = NaN with
  mean FC = 0). dt = 0.05 is still unstable over 600 s; the 45 s diagnostic was too short to show it. The amendment's
  "stable at dt = 0.05" was wrong for long runs.
- Descriptive only (post hoc, not evidence): G↑ raises frac_up (0.33 → 0.61) and η↓ lowers it (0.33 → 0.03), so a
  compensating diagonal may exist; the hand guess δη/δG ≈ −r̄ (≈ −0.4) does not match the crude one-sided ratio (≈ −3).
- Needed before any rerun: (1) an integrator that stays finite over 600 s (smaller dt, or a positivity-preserving or
  implicit step), checked over the full length; (2) steps small enough to stay inside the band, or a design along
  diagonals δη = −k δG; (3) a statistic-level approach suited to rare switching (pathwise derivatives will not work
  here). Not rerun; awaiting the user.

## 2026-11-23 (night), ridge check v2 in JAX/vbjax (user: "lets do 1. on vbjax and think about what extra utility jax can provide")
Env `sim/.venv_vbjax` (vbjax 0.0.19, jax 0.6.2 CPU). vbjax's `mpr_default_theta` = (τ 1, I 0, Δ 1, J 15, η −5, cr 1, cv 0) is
`sim/mpr_net.py`'s node exactly. What JAX adds: exact Jacobians (bifurcation structure), jit for small dt, vmap batching,
GPU (tower needs the vbjax CUDA image or WSL: ask first).
**Stage 0, registered before running: the deterministic skeleton.** Equilibria do not depend on delays, so folds are
computed on the undelayed network: F(r, v; G, η) = vbjax `mpr_dfun` with c = (G W r, 0), 76-region connectome, unit mean
in-strength. Continue the all-low branch upward in G and the all-high branch downward (Newton with jax.jacfwd, step 0.002,
stability by the eigenvalues of ∂F), bisect each branch's end to 1e-5, for η ∈ {−5.3, −5.15, −5.0, −4.85, −4.7}.
Hand expectation: η and G enter as η + G(W r)_i, so the low branch ends where the most strongly driven node's η_eff reaches
its fold (slope dη/dG ≈ −r_dn·s ≈ −0.08·s) and the high branch where η + G r_up s falls below it (slope ≈ −r_up·s ≈ −1·s).
S0-1: at η = −5.0 the operating point G = 1.05 lies between the two branch ends (bistable band). Prior 0.6 (noise-driven
excursions from a monostable low state would also give switching).
S0-2: the two fold curves are not parallel: slope ratio outside [0.5, 2]. Prior 0.65.
Consequence registered now for stage B (stochastic): if S0-2 holds, up-dwell and down-dwell statistics carry two different
combinations of (G, η) (Kramers: escape rates set by distance to the respective fold), predicting SEPARATE for statistics
that resolve both dwell times, with each dwell statistic's level sets near-parallel to its fold curve; statistics that see
only one of them (or neither) can still sit on a ridge.
Amendment before any fold was located: continuing in G failed because the all-high branch does not end for G ∈ (0, 3)
(the isolated node is itself bistable at η = −5: both R_DN and R_UP are equilibria at zero input). Continuation is done
in η at fixed G ∈ {0.95, 1.0, 1.05, 1.1, 1.15} instead (low branch upward in η, high branch downward); same fold curves,
slopes reported as dη/dG. Predictions unchanged.
**Stage 0 result, first pass (2026-11-23).** Low branch ends at η = −3.377, −3.388, −3.400, −3.412, −3.423 for G = 0.95–1.15
(dη/dG = −0.231; set by node 21, the largest in-strength 1.865; hand guess −r_dn·s ≈ −0.15). S0-1 HELD: η = −5 lies
inside (−5.74, −3.40). High branch: ends at η = −5.74353 for every G, set by node 37, whose in-strength is 0 (it receives
no input, so its high state vanishes at the isolated-node fold whatever G is). Degenerate: the "all nodes high" criterion
tied the branch to an uncoupled node. S0-2 not judged on it. **Amendment before recomputing:** the high branch is the
branch continued from all-high with nodes of zero in-strength left free (criterion: every node with in-strength > 0 high).
**Second pass (same amendment) still gave η = −5.74353 at every G**: the high branch still started with the zero-input
nodes 37 and 75 high, and ended at their isolated fold; the "node moving most" near that point is the weakest-input node
(47, in-strength 0.053). Diagnosis: a single node's fold is the wrong object for global switching. **Second amendment,
registered before computing:** replace branch ends by a quasi-static hysteresis sweep of the undelayed deterministic
network (vbjax node, RK4 dt = 0.01, relax 1000 time units per step, η step 0.02, G ∈ {0.95, …, 1.15}): from the all-low
state sweep η upward and record η_ignite(G) where the attractor's fraction of nodes up (r > 0.5) first exceeds 0.5; from
the high cluster (every node with in-strength > 0 up, zero-input nodes low) sweep η downward and record η_collapse(G)
where it first falls below 0.2. S0-1′: η_collapse(1.05) < −5.0 < η_ignite(1.05). Prior 0.7. S0-2′: the slopes dη/dG of
the two curves differ by a factor > 2. Prior 0.6 (hand: ignition is driven through r_dn·s, collapse through r_up·s).
**Implementation check of `sim/ridge_v2_sde.py` (JAX) against `sim/mpr_net.py` (σ = 0, same initial state).** Neural states
agree to 1e-15 (float64). BOLD first differed by 4.7e-4 relative; cause found in `mpr_net.Balloon.step`: `df = self.s`
aliases the array that `self.s += …` then updates in place, so f is advanced with the NEW s (a semi-implicit Euler step,
by accident). All earlier runs used that version; it is a valid scheme at 1 ms with these constants, so no earlier result
is affected beyond the discretisation choice. The JAX port reproduces it deliberately.
**Stage A, registered before running (GPU, vbjax CUDA image on the tower, float32): numerics.** Centre (G, η, σ) =
(1.05, −5.0, 0.30), seeds 1–8, 600 s, dt ∈ {0.05, 0.025, 0.0125}; `sim/ridge_v2_sde.py`. A1: finite throughout in 8/8
seeds. A2: mean fraction-up and debounced transitions per 600 s agree between dt and dt/2 within 2 standard errors
across seeds. The working dt is the largest that passes A1 and A2 against its half. A3 (reported whatever it shows):
whether the paper-like regime (≥ 3 debounced transitions in most seeds) survives at the converged dt. If it does not,
the operating point found earlier was a discretisation artefact of dt = 0.1/0.05, and stage B would need a new
operating point. Prior that the regime survives: 0.6.
**Stage 0 outcome (hysteresis sweep, `runs/ridge_v2_hysteresis.json`).** Global ignition from the all-low state at
η = −3.36, −3.38, −3.38, −3.40, −3.42 and collapse of the high cluster at η = −6.36, −6.40, −6.44, −6.48, −6.50 for
G = 0.95–1.15. Slopes dη/dG: ignition −0.28, collapse −0.72 (ratio 0.39). Resolution: η step 0.02 over ΔG = 0.2 gives
about ±0.1 on each slope, so the ratio lies roughly in 0.25–0.55. S0-1′ HELD (−6.44 < −5.0 < −3.38). S0-2′ HELD as
registered (ratio outside [0.5, 2]), with the upper end of its uncertainty near the boundary.
Reading: the operating point sits deep inside a wide deterministic bistable band, about 1.4 in η from collapse and 1.6
from ignition. So the switching is noise-driven across barriers, consistent with the v1 finding that ±0.1 in η moves
frac_up tenfold (Kramers rates depend exponentially on barrier/σ²). Caveat for stage B: "level sets parallel to the fold"
is a near-fold approximation, and the operating point is not near either fold; the registered consequence (two
different combinations for up-dwell and down-dwell) is a heuristic there, not a derivation.
**Stage B, registered before stage A's results are seen (dt = stage A's working dt; not run if A3 fails).** Grid
G ∈ 1.05 + {−0.020, −0.015, …, +0.020} (9) × η ∈ −5.0 + {−0.060, −0.045, …, +0.060} (9), σ = 0.30, seeds 1–8 with common
random numbers across all 81 configurations, 600 s, two GPU jobs (split by G). Statistics (`sim/ridge_v2_analyse.py`):
frac_up, debounced transitions, up-dwell, down-dwell, switching index, VAF2, mean FC, corr(FC, SC).
Primary (decides the question): over configurations where the regime holds (≥ 5 of 8 seeds with ≥ 3 transitions), fit
each statistic's seed mean by a quadratic surface in (G, η) weighted by its standard error; take the gradient at the
centre; whiten by the centre's across-seed sd; c = |cos(j_G, j_η)| over the statistics, bootstrap over seeds (resample
seeds, refit; 2000). RIDGE if c > 0.95 and 5th percentile > 0.9; SEPARATE if c < 0.8 and 95th percentile < 0.9; else
INCONCLUSIVE. Priors (after stage 0): SEPARATE 0.5, RIDGE 0.25, INCONCLUSIVE 0.25. Validity: fewer than 9 configurations
with the regime → report, do not score.
Secondary (stage 0 heuristic): level-set slopes dη/dG = −(∂_G)/(∂_η) of up-dwell and down-dwell; prediction: up-dwell's
nearer the collapse slope (−0.72), down-dwell's nearer the ignition slope (−0.28). Prior 0.5. Descriptive: the shape
of the region where the regime exists.

## 2026-11-23 (night), structural identifiability of the MPR network from its equations (user: "Run alongside"; registered before running)
Pre-data probe: no noise, no statistics. Undelayed deterministic network of vbjax `mpr_dfun` nodes on the 76-region
connectome; parameters θ = (G, η, J, Δ, τ, w), w = scale of the connectome (coupling G·w·W r); float64; `sim/ridge_v2_structural.py`.
⟦HAND⟧ found while planning: the equations have a hidden-coordinate scaling symmetry. With v ↦ λv and
(τ, Δ, η, J, G) ↦ (λτ, λ²Δ, λ²η, λJ, λ²G), both MPR equations are satisfied with r(t) unchanged (delays do not matter:
only r is delayed). So for a record of rates with v hidden there are two exact fibre directions, in log-parameters:
n1 = (1, 0, 0, 0, 0, −1) (G against the connectome scale) and n2 = (2, 2, 1, 2, 1, 0) for (G, η, J, Δ, τ, w).
Predictions:
T1 (global, exact): r(t) from θ and from the λ = 1.3 scaled θ with v₀ ↦ λv₀ agree to < 1e-10 over 50 ms. Prior 0.95.
T2 (record = r at 200 times × 76 nodes over 50 ms from a perturbed mixed start; hidden start v₀ as nuisance): the
relative-sensitivity Jacobian has exactly 2 singular values below 1e-9·σ_max, null space = span(n1, n2) (principal
cosines > 0.999999), and a gap ≥ 1e3 to the third smallest. Prior 0.8 (a further symmetry, or conditioning, would defeat it).
T3 (v₀ known): exactly 1 such null direction, n1; n2 is seen through the transient (above 1e-9). Prior 0.85.
T4 (TVB practice: J, Δ, τ, w fixed; only G, η free; v₀ nuisance): full rank, σ_min/σ_max > 1e-3. Prior 0.85.
T5 (record = an equilibrium's rates, hence steady BOLD; by the implicit function theorem; all-low state and the high
cluster): 2 singular values below 1e-9·σ_max = span(n1, n2), gap ≥ 100 to the third; (G, η) block ratio > 1e-3.
Prior 0.7 (near-collinear regressors across nodes may blur the gap in the low state).
Scope: equations → record only. Says nothing about noise, delays' effect on conditioning, or the fitted statistics.
**Outcome (2026-11-23, night; `sim/ridge_v2_structural.py`).**
- T1 HELD: max |r − r_scaled| = 3.1e-15 over 50 ms (control without rescaling v₀: 7.3e-2). The scaling symmetry is exact.
- T2: two singular values at roundoff (4.9e-16, 2.4e-17 of the largest), the next at 1.9e-2: count and gap HELD.
  **Null-vector criterion FAILED as registered**: my n2 = (2, 2, 1, 2, 1, 0) had the wrong sign in the η coordinate
  (columns were scaled by |θ| and η < 0; the symmetry's tangent in those coordinates is (2, −2, 1, 2, 1, 0)). Post hoc with
  the corrected vector: principal cosines (1, 1), |J n2| = 4.6e-16. A bookkeeping error, not another symmetry (T1 is the
  independent confirmation).
- T3 HELD: with v₀ known, one null direction (n1); the scaling direction is seen through the transient at 2.0e-3.
- T4 HELD: (G, η) alone, σ_min/σ_max = 0.063 (v₀ nuisance), 0.062 (v₀ known); |cos| between the G and η columns 0.87–0.88.
- T5 HELD on count, gap and the (G, η) ratio (same sign correction for n2; corrected cosines (1, 1) in both states).
  All-low equilibrium (rates 0.081–0.083): non-null singular values 1, 7.9e-3, 1.2e-5, 3.3e-9; (G, η) ratio 8.0e-3.
  High cluster (74/76 nodes up): 1, 7.4e-2, 6.9e-3, 1.4e-3; (G, η) ratio 0.108.
Reading. (1) With J, Δ, τ and the connectome scale fixed, as in TVB practice, G and η are structurally identifiable
from a record of rates: the equations are not degenerate in those two. A ridge in stage B, if found, belongs to the
statistic, the noise or the regime. (2) With all node parameters free there are exactly two fibre directions at this
point: G against the connectome scale, and the hidden-v scaling (τ, Δ, η, J, G) ↦ (λτ, λ²Δ, λ²η, λJ, λ²G), v ↦ λv, a
change of hidden coordinates in the sense of the note's §5.1 (probably dimensional-analysis folklore for QIF mean
fields; not searched). (3) Conditioning is a separate matter: near the low state, steady rates carry about two usable
combinations of six parameters before any noise. Scope: noise-free, undelayed, one generic point (local) for T2–T4;
"steady BOLD is an invertible function of the steady rate" is ⟦HAND⟧, not checked.
**Stage A, partial (2026-11-23, 23:10), and an amendment registered before the dt = 0.025 and 0.0125 files exist.**
- Correction: both stage A containers ran on GPU 0 (RTX 3060). `--gpus device=N` is not honoured by Docker Desktop on
  the tower (inferred: a container started with device=0 listed both GPUs; GPU 1 stayed at 0% while "its" job ran).
  Stage B containers will set `CUDA_VISIBLE_DEVICES` instead.
- dt = 0.05 (19.3 min): finite in 7/8 seeds (seed 6 non-finite from TR 588 ≈ 423 s). **A1 FAILS for dt = 0.05.**
  frac_up 0.30 ± 0.02, VAF2 0.976, mean FC 0.66, corr(FC, SC) 0.60: the same regime as v1's centre.
- Detector flaw found in this file: the per-TR up-fraction is bimodal with plateaus near 0.08 and 0.58 (95% quantile
  0.57–0.59 in every seed, maximum 0.61–0.64). The registered upward threshold 0.6 sits ON the upper plateau, so the
  registered transition counts (0–4 here; 4–18 in v1 with 100 ms samples) are threshold noise, in v1 as well.
  **Amendment (informed by the dt = 0.05 file only):** state = up when the 3-TR mean (edge-normalised) exceeds 0.45, down
  when it falls below 0.20. A3 and stage B's "regime holds" rule and dwell times use this detector. v1's transition
  counts are withdrawn as a description of the regime (its frac_up, VAF2 and FC values stand).
  dt = 0.05 re-scored with the amended detector (post hoc for this file): 6–27 transitions per 600 s (mean 20.6),
  up-dwell 28 ± 4 s, down-dwell 45 ± 18 s (seven finite seeds). Differs from the dt = 0.1 description in the ledger
  (up 10–13 s, down 106–137 s); whether that is dt or the detector is what A2 will show.

## 2026-11-24 (00:30), stage A scored: the statistics do not converge in dt; stage B NOT RUN (registered gate)
Files `sim/runs/ridge_v2/stageA_dt{0.05,0.025,0.0125}.npz`; `sim/ridge_v2_analyse.py stageA`; amended detector. Mean ± se
over finite seeds (8 seeds, 600 s, centre (G, η, σ) = (1.05, −5.0, 0.30); JAX, float32, tower GPU 0; wall 19, 63, 145 min, shared GPU).

| dt (ms) | finite | frac_up | transitions / 600 s | seeds with ≥ 3 | VAF2 | mean FC | corr(FC, SC) |
|---|---|---|---|---|---|---|---|
| 0.05 | 7/8 | 0.296 ± 0.017 | 20.6 ± 2.6 | 7/7 | 0.976 ± 0.001 | 0.657 ± 0.009 | 0.603 ± 0.014 |
| 0.025 | 8/8 | 0.564 ± 0.002 | 2.6 ± 0.8 | 5/8 | 0.890 ± 0.018 | 0.461 ± 0.033 | 0.692 ± 0.019 |
| 0.0125 | 8/8 | 0.590 ± 0.001 | 0.4 ± 0.3 | 0/8 | 0.607 ± 0.055 | 0.264 ± 0.022 | 0.733 ± 0.010 |

- **A1:** fails at dt = 0.05 (7/8); holds at 0.025 and 0.0125.
- **A2 FAILS for both pairs.** 0.05 vs 0.025: frac_up differs by 0.268 (16 combined se), transitions by 17.9 (6.6 se).
  0.025 vs 0.0125: frac_up by 0.026 (10.8 se), transitions by 2.25 (2.6 se); VAF2 and mean FC also move (0.89 → 0.61,
  0.46 → 0.26). **No working dt:** none passes A1 and A2 against its half; 0.0125 has no half-step run, so it is untested.
- **A3:** there is no converged dt to score it at. Reported as registered: ≥ 3 transitions in 5/8 seeds at 0.025 and in
  0/8 at 0.0125. At the finest step the network stays on the upper plateau (frac_up 0.59) for the whole record.
  Registered prior that the regime survives was 0.6; it did not survive to dt = 0.0125.
- Dwell times at 0.025 (up 304 s over 5 seeds, down 7 s) are censored by the 600 s record and by the seeds with no
  transition; not estimates. At 0.0125 there are none to report.
- **Stage B is not launched**: its registered conditions (a working dt from stage A; A3) are not met. The 5/8 at dt = 0.025
  does not license it, since 0.025 fails A2.
- **Scope of the evidence.** The dt trend is seen through one code path (the JAX simulator, float32). Its stochastic
  path agrees with `mpr_net.py` statistically at dt = 0.05 only (frac_up 0.30 vs 0.31–0.34, VAF2 0.976 vs 0.977–0.979)
  and deterministically at σ = 0. No independent check at dt < 0.05 yet.
- **Candidate reversal, PENDING a reversal check; nothing propagated.** Everything characterised at (G, σ) = (1.05, 0.30)
  was run at dt = 0.05: gate G0e and its re-scoring, G1 in the transition regime, the rewiring gate
  (`g2c_rw{11,12}_trans`), the two-state reduction (`g3_trans`) in `notes/sfm_statistic_indexed_critique.md`; ridge
  check v1's descriptive numbers; panel B of `notes/figures/fig2_tvb_draft`. If the trend is confirmed, those describe
  the dt = 0.05 scheme, not the stochastic delay equation. Those entries are not edited yet.
  Unaffected (noise-free): stage 0 (hysteresis sweep) and the structural identifiability probe.
- The paper's text (`essay/v3/audit/fousek2024_fulltext.txt`, Methods) gives the equation, "implemented in The Virtual
  Brain", and the working point G = 0.540, σ = 0.030 (its own noise convention and connectome); a search of that text for
  the integration step or scheme finds nothing. Its dt would have to be looked up in its code or supplement. Not done.
- Tower: finished containers `sfmA_dt0125`, `sfmA_dt05_025` removed.

## 2026-11-24 (00:45), why the statistics move with dt: one-node probe (registered before running)
Smallest case: ONE uncoupled MPR node (η = −5, vbjax defaults) with a constant input I in place of the coupling; same
scheme as `sim/ridge_v2_sde.py` (Heun drift, Euler noise σ = 0.30 on r and v, r ← max(r, 0)); float64, laptop.
dt ∈ {0.05, 0.025, 0.0125, 0.00625}; I ∈ {0.2, 0.65} (⟦HAND⟧: mean input to a node of unit in-strength on the low and
the high network plateau); starts at the down and the up fixed point; 256 replicates; 2000 ms, first 200 discarded.
Variants: (a) as used; (b) no clamp, r may go negative; (c) noise on v only (the clamp cannot bind: ṙ = Δ/π > 0 at
r = 0). Script `sim/ridge_v2_node_dt.py`. Measured per cell: clamp-binding fraction (steps with r < 0 before the
clamp), mean r, mean r while down (r < 0.5), up-occupancy (fraction of time r > 0.5), 99.9th percentile of r, finite.
Hypothesis under test (⟦HAND, unverified⟧): the clamp is a projected boundary. Per-step noise on r has sd σ√dt = 0.067,
0.047, 0.034, 0.024, against a down-state rate of 0.081 with stationary sd ≈ σ/√(2·|2v|) ≈ 0.107, so the clamp binds
often. A projected boundary converges like √dt, not dt, and acts like a reflecting wall displaced to about −0.58 σ√dt
(Asmussen–Glynn–Pitman 1995, from memory, not re-read): coarse steps lower the down-state rate, hence the input to the
other nodes. Rough size (v frozen at its fixed point, Gaussian low state): down-state mean r ≈ 0.107, 0.111, 0.114,
0.116 at the four step sizes, 0.122 in the limit.
- N1 (binding): variant (a), down start, I = 0.2: the clamp binds on more than 3% of steps at every dt, falling by a
  factor between 1.2 and 1.7 per halving. Prior 0.8.
- N2 (dt-dependence of the output): same cells: mean r while down rises monotonically as dt falls, successive
  differences shrinking by a factor between 1.2 and 1.7 per halving. Prior 0.6.
- N3 (control): in (b) and (c) the same quantity changes between dt = 0.05 and 0.00625 by less than one fifth of its
  change in (a). Prior 0.7.
- N4 (does it reach the regime?): in (a), up-occupancy at fixed input differs between dt = 0.05 and 0.0125 by more than
  3 se in at least one (I, start) cell, upward as dt falls, and not in (b). Prior 0.4. My own size estimate says the
  mean-rate shift alone is too small: ΔI ≈ G·Δr ≈ 0.007, against the v1 sensitivity of about a factor e per 0.04 in η.
  Not evaluable if no replicate changes state in any cell (then switching in the network needs input fluctuations).
Defeat of the clamp hypothesis: N1 fails (binding rare), or N3 fails (the trend is as large without the clamp: then the
cause is the scheme's resolution of large excursions, candidate two), or N4 fails with N2 holding (the clamp biases the
rate but one node at fixed input does not reproduce the regime shift: the cause is collective and the question moves
back to the network). Not tested here: delays, input fluctuations, in-strength heterogeneity.
**Outcome (2026-11-24, 01:10; `sim/runs/ridge_v2/node_dt.json`; 50 s, 0.29 GB).** All cells finite 256/256. The two
starts agree in every (a) and (b) cell, so those occupancies are stationary. Deviation from the registration: the script
records the maximum of r, not its 99.9th percentile (used in no prediction).
- **N1 HELD.** Binding on 11.4%, 8.4%, 6.1%, 4.5% of steps; ratio per halving 1.37, 1.37, 1.38 (√2 = 1.41).
- **N2 FAILED in its rate part.** Mean r while down rises monotonically as dt falls (0.1353, 0.1368, 0.1382, 0.1399;
  se 0.00016), but the successive rises do not shrink (0.0015, 0.0015, 0.0017; ratios 1.03, 0.84; registered 1.2–1.7).
  No levelling off by dt = 0.00625. The hand level (0.107–0.116) was also low by about a quarter. Unexplained.
- **N3 HELD.** Change of the same quantity between dt = 0.05 and 0.00625: (a) +0.0047 ± 0.0002; (b) +0.0004 ± 0.0003;
  (c) 0.0000.
- **N4 HELD as registered, narrowly, and its "not in (b)" clause is weak.** (a), I = 0.65: up-occupancy +0.018 from
  dt = 0.05 to 0.0125 (3.07 se, down start; 2.89 se, up start); all four (b) cells within 2 se. Size: 0.430 → 0.448.
  "Above 3 se in (a), below 2 se in (b)" is not a difference between (a) and (b): the direct contrast at I = 0.65 is
  about 1.5 se on occupancy and about 2 se on mean output.
- **Attribution: N3 was scored only on the down-state rate at I = 0.2.** For the mean output at I = 0.65, on the step
  where the network moved most (dt = 0.05 → 0.025), the rise is about as large without the clamp (+2.6%) as with it
  (+3.0%). The registered defeat condition ("as large without the clamp") is partly met there. So: two contributors;
  which one carries the network's jump between 0.05 and 0.025 is unresolved.
- Node output at fixed input (mean r, starts pooled; dt = 0.05, 0.025, 0.0125, 0.00625).
  (a) I = 0.2: 0.1854, 0.1886, 0.1903, 0.1936 (+4.4%); I = 0.65: 0.5395, 0.5554, 0.5580, 0.5639 (+4.5%); se 0.001, 0.003.
  (b) I = 0.2: 0.1458, 0.1468, 0.1451, 0.1451 (flat); I = 0.65: 0.4794, 0.4917, 0.4886, 0.4913 (+2.6% between 0.05 and
  0.025, 2.9 se, then flat).
Reading (post hoc; the link to the network is ⟦HAND⟧, not checked through the network):
1. The clamp makes the node's output depend on dt, and the trend has not levelled off at dt = 0.00625. Without the
   clamp that trend is absent. So with this scheme a further halving of dt would not have rescued A2.
2. A second contribution is not the clamp: between dt = 0.05 and 0.025, at high input, the output rises about 2.6% with
   or without it (one comparison, 2.9 se). Candidate: dt = 0.05 under-resolves the large excursions (v1's diagnostic
   had peak rates growing as dt fell).
3. Size. Scaling every node's output by 1 + x acts on the mean input like scaling G by 1 + x. Ridge check v1
   (descriptive, one-sided, dt = 0.05) had frac_up 0.33 → 0.61 for G + 3%. The node's output rises 1.7–3% from
   dt = 0.05 to 0.025 and 2.6–3.4% to 0.0125; the network went 0.30 → 0.56 → 0.59. The sizes are compatible, not
   shown: the comparison rests on v1's one-sided G + 3% number, from runs with non-finite seeds. My registered estimate
   that the shift was too small was wrong: it counted the down-state rate only, not the whole output.
4. Not registered, and larger than the dt effect: the noise on r sets the regime, and the boundary rule adds to it.
   Down-state rate 0.084 (noise on v only), 0.102 (no clamp), 0.135–0.140 (clamp). With noise on v only, no replicate
   left the down state in 256 × 1.8 s at either input. With noise on r, a node at I = 0.65 is up 39–40% of the time
   without the clamp and 43–45% with it. So ignition is driven by the noise on the rate, whose per-step size is
   comparable to the down-state rate itself; the clamp adds a few points on top.
**What this says.** The simulated object is the equations plus a boundary rule plus a step size. At σ = 0.30 on r the
last two move the node's output by as much as a 3% change in G, and the operating point's band is narrower than that.
**What it does not say.** That this reaches the paper. σ = 0.30 on both variables is this folder's convention
(`sim/mpr_net.py`: "not mapped exactly"); the paper's working point is σ = 0.030 in TVB's convention; its step size, its
noise on r and TVB's boundary handling have not been looked up. The paper's text names no code repository (simulated
data "on reasonable request"; a supplement exists, not read for this). Scope: one uncoupled node, constant input, two
input values, one seed of the noise generator.
**Open, the user's choice (nothing started):** (i) look up the paper's step size, noise convention and TVB's boundary
rule (supplement or TVB source), which decides whether any of this touches the paper; (ii) the reversal check: the
network at dt < 0.05 through the NumPy path (`mpr_net.py`), independent of the JAX simulator; (iii) a new operating
point under a scheme that converges (a decision about the boundary first), then stages A and B again.

## 2026-11-24 (05:15), reversal check of the dt trend through the NumPy path (user: "Proceed"; registered before running)
Claim under check: at (G, η, σ) = (1.05, −5.0, 0.30) the network's statistics depend on the step: switching with
frac_up ≈ 0.30 at dt = 0.05, mostly up (≈ 0.56) at 0.025, up throughout (≈ 0.59) at 0.0125. So far seen through
`sim/ridge_v2_sde.py` only (JAX, float32, tower GPU).
Independent route: `sim/mpr_net.py` `run_batch`, unchanged (NumPy, float64, its own generator and initial state, the
code all the earlier gates used), driven by `sim/ridge_v2_numpy_check.py`; tower CPU, one process per run, 600 s;
same statistics and amended detector (`ridge_v2_analyse.stats_one`); frac_up from r kept every 10 ms.
Runs: dt = 0.05, seeds 1–2; dt = 0.025, seeds 1–4; dt = 0.0125, seeds 1–2 (all with the half-up random start);
dt = 0.025, seeds 1–2 from the all-down start (hostile to "mostly up is an effect of starting half up").
- RC1 (control): dt = 0.05: frac_up in 0.20–0.40 and ≥ 3 transitions in both seeds. If it fails the comparison is
  invalid (driver or detector), not a result. Prior 0.85.
- RC2 (the reversal): dt = 0.025: frac_up > 0.50 in at least 3 of 4 seeds. Prior 0.85. **Defeat:** frac_up ≤ 0.40 in
  most seeds: the trend belongs to the JAX simulator (precision, noise generation or a bug), stage A is void and the
  candidate reversal is withdrawn.
- RC3: dt = 0.0125: frac_up > 0.55 and fewer than 3 transitions in both seeds. Prior 0.8.
- RC4 (start): dt = 0.025 from all-down: frac_up after the first 100 s > 0.50 in both seeds. Prior 0.75. If it fails,
  both plateaus are long-lived at fine steps and "mostly up" in stage A reflects its start; the step-dependence
  would stand (0.05 switches, 0.025 does not) but its description changes.
Then, as the programme's rule for reversals requires: a fresh hostile agent reads the records and writes
`notes/OPERATING_POINT_DT_REVERSAL_CHECK.md`. Nothing is propagated before that.
**Outcome, first eight runs (2026-11-24, 06:20; `sim/runs/ridge_v2/rc/rc_dt*.npz`; `ridge_v2_numpy_check.py score`).**
All finite (833/833 TRs). Wall 29 min (dt = 0.05) and 53 min (0.025) per run, ten processes on six cores.

| run (NumPy path) | frac_up | transitions / 600 s | VAF2 | mean FC | JAX stage A, same dt |
|---|---|---|---|---|---|
| dt 0.05, seeds 1, 2 | 0.279, 0.288 | 20, 13 | 0.976, 0.978 | 0.63, 0.70 | 0.296 ± 0.017; 20.6; 0.976; 0.66 |
| dt 0.025, seeds 1–4 | 0.555, 0.562, 0.576, 0.563 | 2, 1, 0, 2 | 0.93, 0.88, 0.83, 0.91 | 0.44, 0.35, 0.33, 0.48 | 0.564 ± 0.002; 2.6; 0.89; 0.46 |
| dt 0.025, all-down start, seeds 1, 2 | 0.560, 0.562 (after 100 s: 0.564, 0.570) | 3, 1 | 0.89, 0.83 | 0.41, 0.36 | — |

- **RC1 HELD** (control): 0.279 and 0.288, 20 and 13 transitions.
- **RC2 HELD: the step-dependence reproduces through the independent path.** frac_up > 0.50 in 4/4 seeds at dt = 0.025.
  The defeat condition (a JAX artefact) is not met.
- **RC4 HELD:** from the all-down start the network is mostly up after 100 s in both seeds. "Mostly up" at dt = 0.025 is
  not an effect of the half-up start.
- RC3 (dt = 0.0125, two runs): still running; appended below when they finish.
- **RC3 HELD (2026-11-24, 07:05; 87 min per run).** NumPy path, dt = 0.0125, seeds 1 and 2: frac_up 0.592 and 0.593;
  1 and 0 transitions; share of TRs below 0.3: 0.5% and 0%; plateau 0.598 and 0.599 (JAX: 0.590 ± 0.001; 0–2;
  0–0.5%; 0.596–0.601). The finest step is now seen through both paths. All four registered predictions held.
**Hostile check returned (2026-11-24, 06:34): `notes/OPERATING_POINT_DT_REVERSAL_CHECK.md` (fresh opus agent,
reviewed: false). All three claims CONFIRMED WITH NARROWER WORDING.** I spot-checked its detector and low-time table
against the stage A files: it reproduces. Its own node reruns I did not rerun.
- **The step-dependence stands, and rests on detector-free numbers.** frac_up (0.30 → 0.56 → 0.59), the share of TRs
  with frac_up < 0.3 (39–67% → 0.2–3% → 0–0.5%), and the plateau level (0.55 → 0.586 → 0.597). Transition counts cannot
  carry it: under the registered 0.6/0.4 rule dt = 0.05 has fewer transitions (1–4) than dt = 0.025 (2–16).
- **Corrections to my statements.** (1) dt = 0.025 has 0–6 transitions per seed (JAX), not 0–3; dt = 0.0125 has 0–2,
  not none. (2) The partial-stage-A note above ("differs from the dt = 0.1 description in the ledger … whether that is
  dt or the detector") is wrong: G0e ran at dt = 0.05; the difference is the detector. Re-scored with the amended rule
  the G0e records give 22 and 22 transitions, up ≈ 26 s, down ≈ 29 s, frac_up 0.29 and 0.34. So the description "6–8
  transitions, high 10–13 s, low about two minutes" is a detector artefact at dt = 0.05 itself. (3) "The rise is about
  as large without the clamp (+2.6%) as with it (+3.0%)" does not reproduce: two paired redraws give about half
  (+0.008 ± 0.002 against +0.014 ± 0.002, at I = 0.65 only; absent at I = 0.2). The clamp's contribution grows at every
  halving down to dt = 0.003125. (4) "No step converged" hides structure: in dip-free 60 s windows VAF2 and mean FC
  agree between 0.025 and 0.0125; only the rate of brief collective dips has not converged. (5) The object at dt → 0,
  if the scheme converges, is a reflected equation, not the paper's formula as written.
- **Scope, per entry (its §3).** Claims that a regime EXISTS at or near (1.05, 0.30) belong to the dt = 0.05 scheme:
  G0e, the band and boundary claims, the rewiring contrast, ridge v1's band, Figure 2 draft panel B, status lines 37
  and 40 of this file, essay v3 lines 210, 237, 255, 323, 351, 413. Claims about statistics computed ON those records
  (G1 against both nulls; the two-state rival `g3_trans`) keep their logic and need only the generator relabelled;
  `g3_trans`'s "2 vs 6–8 transitions" becomes 1 vs 22 under the amended rule. Unaffected: stage 0, the structural probe.
- **Hole it names:** every simulated gate in the ledger ran at dt = 0.05; whether the other regime labels move with the
  step is untested.
- **Propagation: NOT yet done.** The places are listed in its §3 and §5. The ledger and this file take dated
  correction notes; essay v3 and the figures are the user's text and wait for a decision.

## 2026-11-24 (05:45), lookup: the paper's step size and noise, and TVB's boundary rule (one sonnet scout; user: "Proceed")
Read depth is stated per item. "Source read" = I downloaded the file and read the lines myself (GitHub, master branch,
2026-11-24). "Scout" = returned by the scout through a summarising fetch tool, not re-read by me.
- **The paper states neither.** Main text (read, `essay/v3/audit/fousek2024_fulltext.txt`): "implemented in The Virtual
  Brain"; no step, no scheme, no split of the noise between r and V. Supplement: figures S1–S7 only (scout, read as page
  images). Preprint v2: nothing found (scout; the tail of the page was not read). No public code found; simulated data
  "on reasonable request".
- **TVB's rule is the same projection** (source read). `MontbrioPazoRoxin.state_variable_boundaries` = {"r": [0, inf]}
  (`tvb/simulator/models/infinite_theta.py`, lines 128–131). `_bound_state` sets a value below the lower bound to the
  bound (`integrators.py`, 181–186). `HeunStochastic.scheme` applies it twice: to the predictor
  (`inter = X + dt*m_dx_tn + noise + …`; `integration_bound_and_clamp(inter)`) and to the result
  (`X_next = X + dX + noise + …`; `integration_bound_and_clamp(X_next)`). The predictor carries the noise; in this
  folder's scheme it does not. Default dt of a TVB integrator: 0.01220703125.
- **TVB's noise convention** (source read, `noise.py`): `white` returns `sqrt(dt) * normal`; `Additive.gfun` returns
  `sqrt(2.0 * nsig)`. Per-step sd = √(2·nsig·dt). So this folder's σ (sd per √ms) = √(2·nsig).
- **vbjax** (source read, installed 0.0.19): `mpr_r_positive` returns `r*(r>0)`, offered as the `adhoc` correction and
  applied by `heun_step` to the predictor and to the result. The same projection.
- **Sibling work of the same group** (not the paper). `ins-amu/virtual_aging_brain`, `src/simulation.py` (source read):
  `dt=0.01`, `HeunStochastic`, `nsig=np.r_[noise_sigma/tau, noise_sigma*2]` with `tau = 1`, default `noise_sigma=0.01`.
  Gudibanda, Fousek, Petkoski, Jirsa 2025 (bioRxiv 10.1101/2025.02.10.637533; scout only): stochastic Heun,
  "J = 15, η = −5, Δ = 1 and dt = 0.01ms", noise on both variables with "σv = 2σr", levels 0.014, 0.037, 0.051.
  Rabuffo et al. 2021 (scout only): "Dt = 1 ms" as the resolution of the simulated signal; no scheme, no noise amplitude.
**What the lookup settles.** (1) The projection at r = 0 is the simulators' own rule (TVB and vbjax), not this
folder's invention. (2) The sibling practice is a step of 0.01 ms, five times finer than the 0.05 at which this folder's
operating point was found, and between the two finest steps of stage A, where that point does not switch.
**What it leaves open, and it decides whether the boundary matters in the paper's runs** (⟦HAND⟧ arithmetic):
the paper's σ = 0.030 has two readings a factor 8 apart. (a) As written in its Methods (increment "with variance σ²"):
sd 0.03 per √ms. The down-state rate 0.081 then has stationary sd ≈ 0.011 and sits about 7 sd from zero: the clamp
would almost never bind. (b) As TVB's `nsig` with the siblings' split [σ, 2σ]: sd √0.06 = 0.245 on r and √0.12 = 0.346
on V, close to this folder's 0.30 on both, where the clamp binds on 4–11% of steps. Nothing public says which.
Not established: that any result of the paper depends on its step. Under reading (b) the paper's runs would be in the
regime where the boundary rule is part of the model; under (a) they would not.

## 2026-11-24 (06:45), lookup, second pass (user: "that needs to be reported somewhere … possibly in supplementary material?")
Everything in this entry I downloaded and read myself (full-text XML or source files, 2026-11-24), unless marked.
**The paper reports neither, in any version.**
- Journal supplement (`41598_2024_83542_MOESM1_ESM.pdf`, 7 pages): Figures S1–S7 with captions, manuscript lines
  1193–1243. No Methods text.
- bioRxiv v1 (2022-01-04) and v2 (2023-04-20), full-text XML: the same Methods sentences as the journal version; no
  step, no scheme, no split of the noise; no supplementary file attached. v1 defers the node parameters to Rabuffo et
  al. 2021 (the journal version cites Montbrió 2015 there).
- Rabuffo et al. 2021 (eNeuro, PMC8555887): "numeric integration … using TVB"; sampling rate of the state variables
  1000 Hz; "the resolution Dt of the simulated signal, which physical interpretation is arbitrary. We impose it to be
  Dt = 1 ms". No integration step, no noise amplitude in the text.
- No repository for the paper: none in the `ins-amu` organisation (44 repositories listed) or in the first author's
  account (38), none by repository search.
**The closest public code states both.** `ins-amu/PCI_Fluidity`, the code of Breyton, Fousek, Rabuffo, Sorrentino,
Kusch, Massimini, Petkoski, Jirsa (eLife 2025, 10.7554/eLife.98920). Why it is close: 84 cortical and subcortical
regions; η = −5, J = 15, Δ = 1; the same fluidity statistic; its optimum G = 0.56, σ = 0.036 sits next to the paper's
G = 0.540, σ = 0.030; its sweep grid (G 0.25–0.70 step 0.01; nsig 0.020–0.058 step 0.002) contains the paper's working
point on grid points; `utils/simulation.py` holds the helpers of the paper's escape-time analysis (a time-reversed MPR
model for the separatrix, fixed points). The connectome shipped there (`data/external/hcp-001.zip`) has 84 regions.
- `snakefiles/resting-state-simulations.snake`: `DT = 0.01`; `FULL_SIM = 60000`; `conn_speed=20.`; `tavg_t_bf *= 10`.
- `utils/simulation.py`, `configure_sim`: `integrators.HeunStochastic(dt=dt, noise=noise.Additive(nsig=np.array(
  [nsigma, nsigma*2]), …))`; `monitors.TemporalAverage(period=period)` with the comment "we rescale time later, to
  'slow down' the system"; weights divided by their maximum.
- It runs TVB's `NbMPRBackend`. Template `nb-montbrio.py.mako` (tvb-root, lines 145–159): predictor
  `x + dt*dx_0 + noise + stim`, then the bound; corrector `x + dt*(dx_0 + dx(int))/2 + noise + stim`, then the bound.
  The noise buffer is `noise.generate(...) * gfun` = √dt·N(0,1)·√(2·nsig) (`nb_mpr.py`, lines 134–136).
- That paper's text calls σ the "standard deviation" of the noise while its code passes σ as `nsig`. So the gap
  between text and code over what σ means exists in the group's own companion paper.
- A second public code of the group, `ins-amu/virtual_aging_brain` (`configure_default_sim`): `dt=0.01`, Heun
  stochastic, `nsig=np.r_[nsigma, nsigma*2]`, weights divided by their maximum; but `conn_speed=2.0`,
  `tavg_period=1.` and Rabuffo's node parameters. The two codes agree on step, scheme and split; they differ on the
  time convention.
- Gudibanda, Fousek, Petkoski, Jirsa 2025 (bioRxiv, full-text XML read): "stochastic Heun method"; "σv = 2.σr" at
  0.014, 0.037, 0.051; "J = 15, η = −5, Δ = 1 and dt = 0.01 ms". It cites the paper as earlier work with this model; it
  does not say the paper used these settings. Its stochastic equation is an image and was not read.
**Inference, mine, not a fact about the paper:** it very probably used the same pipeline: step 0.01 (model time units),
stochastic Heun with the noise in both stages, r projected at both stages, `nsig = [σ, 2σ]`. Three public sources of the
same group agree on those four; none contradicts. Not determinable from public sources: the paper's own values, and its
time convention (speed 20 with time × 10, or speed 2).
**What follows if the inference holds** (⟦HAND⟧):
- σ = 0.030 is TVB's `nsig` on r: reading (b) of the entry above. In this folder's units (sd per √time unit): 0.245
  on r, 0.346 on V. The down-state rate 0.081 then has stationary sd ≈ 0.088 and sits about 0.9 sd from zero, and the
  per-step noise on r is 0.0245. The projection at r = 0 is active in the paper's runs.
- This folder's reimplementation differs from that pipeline in: the step (0.05 against 0.01); equal noise on r and V
  (theirs √2 larger on V); no noise in the predictor; no bound on the predictor; weights scaled to unit mean
  in-strength (theirs divided by the maximum: mean in-strength 1.54 on their connectome, so their G = 0.54 is 0.83 in
  this folder's scale); and possibly the time convention (a factor 10 between delays and the node's time constant).
- Not established: that any result of the paper changes with its step. The one-node probe saw a drift still present at
  steps of 0.0125–0.00625 in this folder's variant of the scheme; theirs was not tested.

## 2026-11-24 (07:45), running the group's own code: install, static checks, lineage (user: "lets do it")
Change of order, told to the user: the numerics check goes through THEIR code (the real code path); this folder's JAX
port is deferred until the (G, η) map needs it.
- **Connectome note.** `CLAUDE.md` says this project uses the public 76-region TVB connectome. The runs registered
  below use the companion repository's 84-region file instead, `sim/external/pci_fluidity/hcp-001.zip` (copy of
  `ins-amu/PCI_Fluidity`, commit 2bbe4a3, 2025-10-23, MIT licence; `utils/simulation.py`, `utils/bold.py` and the two
  snakefiles are kept beside it, unchanged). On the tower: `C:\Users\pmahon\sfm_ridge\pci_ref\`. That this is the
  paper's subject is NOT known; the paper says "a subject from the Human Connectome Project … 84 regions".
- **Install (tower, isolated).** `C:\Users\pmahon\sfm_ridge\venv_tvb` (Python 3.12), prebuilt packages only
  (`--only-binary=:all:`; no compiler used): tvb-library 2.10.0, numpy 2.5.5, numba 0.68.1, scipy 1.18.1, Mako 1.4.3,
  shapely 2.2.0.
- **Static checks of what was installed** (files copied back and compared with the master copies read earlier, line
  endings normalised): `backend/nb_mpr.py`, `backend/templates/nb-montbrio.py.mako`, `noise.py`, `integrators.py` are
  identical. `models/infinite_theta.py` differs from master elsewhere; in the installed file `MontbrioPazoRoxin` has
  `state_variable_boundaries = {"r": [0.0, inf]}` and the same two equations. `simulator.py`,
  `_configure_integrator_noise`: an `nsig` of shape (nvar,) is reshaped to (nvar, 1, 1), so `[σ, 2σ]` is applied per
  variable (r: √(2σ·dt), V: √(4σ·dt) per step). Static reading only.
- **NOT yet done: the dynamic checks** (`sim/tvb_ref_verify.py`: rendered kernel, measured sd of the noise buffer per
  variable, r landing exactly on 0, timing). Running that script on the tower was refused twice by the session's
  permission check; I did not try another route. It waits for the user.
- **Lineage (GitHub commit history of tvb-root, read through the public API).** The numba MPR backend and its
  template were written by the paper's first author (account `i-Zaak`, profile name Jan Fousek) between 2021-03-31 and
  2021-09-14; the first preprint is dated 2022-01-04. The FIRST version of the template (commit f68d6fd, 2021-03-31)
  already has `r_bound(r) = r if r >= 0. else 0.` applied to the predictor and to the result, the noise in both
  stages, and an example configuration with `dt = 0.01`, `nsig=np.r_[nsigma, nsigma*2]`, `HeunStochastic`, weights
  divided by their maximum, `conn_speed = 2.0`, Rabuffo's node parameters, and the comment "G=0.720 is too much --
  integration fails numerically (NaN)". So the bound, the step and the split predate the preprint by nine months, in
  the first author's own code. This strengthens the inference of the 06:45 entry; it is still an inference. Note the
  time convention there is speed 2.0 (as in the ageing-brain code), not speed 20 with time × 10.

## 2026-11-24 (08:00), stage A′: numerics at the paper's working point under the group's pipeline (registered before any statistic there is seen)
**Question.** In the group's own pipeline, at the paper's working point, do the statistics move when the step is
halved, and by how much compared with one step of their own grid in G?
**Code path (theirs).** `simulation.configure_conn(hcp-001.zip, conn_speed=20.)`, `simulation.configure_sim(G, nsigma,
conn, dt, seed)`, `simulation.run_nbMPR_backend(sim, simulation_length, chunksize)` from the companion repository,
unchanged, on tvb-library 2.10.0 (`NbMPRBackend`). η = −5, J = 15, Δ = 1; linear coupling a = G; weights divided by
their maximum; stochastic Heun, `nsig = [σ, 2σ]`; temporal average of period 0.1. Driver `sim/tvb_ref_run.py`
(the only change to their defaults: `chunksize` = 1000 time units at every step, so the length divides evenly).
**Assumptions, not facts about the paper:** this connectome; speed 20 with one time unit read as 10 ms (their two
public codes disagree on this); the `[σ, 2σ]` split; this backend.
**Runs.** 61000 time units each, the first 1000 discarded (their `WARMUP`; the paper's "10 seconds"), leaving 60000
(the paper's "10 minutes" under the × 10 reading). Tower CPU, one process per run.
- Step series at the working point (G, σ) = (0.54, 0.030): dt ∈ {0.02, 0.01, 0.005}, seeds 1–8; dt = 0.0025, seeds 1–4.
- Their grid neighbours at dt = 0.01: G = 0.53 and 0.55, seeds 1–8.
- Validity control at dt = 0.01: G = 0.30 and G = 0.80, seeds 1–4.
- Smoke first, at an off-target point (G = 0.30, σ = 0.040; `tvb_ref_verify.py`): timing and finiteness only.
**Measures.** Primary, on the rates (no BOLD convention, no detector): (m1) mean rate over nodes and time; (m2)
up-occupancy, the share of (node, sample) with r > 0.7 (the threshold the group published for this model; 0.5 also
reported); (m3) median and (m4) 5th percentile over TRs of the network up-fraction (TR = 72 time units); (m5) VAF of the
first two principal components of r (the paper's VAF, on r sampled every time unit). Secondary, on BOLD (this folder's
Balloon model on r, one sample = 1 ms, TR = 720 samples): switching index of windowed dFC (the paper's fluidity), mean
FC, corr(FC, SC). Reported, not scored: per-node up-occupancy; non-finite runs.
**Rule.** For each primary measure: D_dt = |m(dt 0.01) − m(dt 0.005)|; D_G = |m(G 0.55) − m(G 0.53)| / 2 (one grid step
of theirs, at dt 0.01); se across seeds for both.
- ADEQUATE at the grid's resolution if D_dt < D_G for all five primary measures, or D_dt is within 2 se of zero.
- STEP-DEPENDENT at the grid's resolution if D_dt ≥ D_G and D_dt exceeds 2 se for at least two primary measures.
- Otherwise MIXED: report which measures.
- Trend, scored separately: for each primary measure, whether it moves monotonically across dt = 0.02, 0.01, 0.005,
  0.0025 with the last increment (0.005 → 0.0025) still beyond 2 se (not levelled off) or not.
- Dips and plateaus are described from the distribution of the per-TR up-fraction (quantiles, a histogram); no
  threshold is carried over from the other connectome.
**Validity.** Not scored, only reported, if any of: a non-finite run at dt = 0.01 at the working point; the working
point is not a mixed state (up-occupancy outside 0.05–0.95 in most seeds); the switching index at G = 0.54 is not
above both controls (G = 0.30, G = 0.80) in the seed means. The paper reports a peak of fluidity near G = 0.54 and VAF
0.41 there (ledger); a pipeline that does not put a mixed, more fluid state at 0.54 is not the paper's, and one of the
assumptions above is wrong.
**Priors.** Validity holds 0.65. Given validity: ADEQUATE 0.45, STEP-DEPENDENT 0.25, MIXED 0.30; a measure still
drifting at the finest pair 0.5. ⟦HAND⟧ behind these: the one-node probe saw roughly 1% of node output per halving
from the projection at steps near 0.01 in this folder's scheme; 1% of node output acts on the mean input like 1% of G
(a size heuristic only), about half of one grid step at G = 0.54; their noise on r is smaller (0.245 against 0.30).
**What each outcome would mean.** ADEQUATE: at their step the pipeline's statistics are stable to better than their
own grid; the step-dependence found in this folder is a property of this folder's coarse step and the earlier
reimplementation, not of their runs. STEP-DEPENDENT: the fitted G at the working point depends on the step at the
resolution they report it. Either way this concerns the companion pipeline under the stated assumptions.
**Horizon.** The batch is: install, verification, this check. No (G, η) map and no port sweeps before stepping back.
**State at 08:20: registered, NOT run.** Scripts on the tower and in `sim/`: `tvb_ref_verify.py` (dynamic checks and
smoke), `tvb_ref_run.py` (one run + statistics), `tvb_ref_batch.py` (the 52 registered jobs, K at a time, logs in
`runs_A/`), `tvb_ref_score.py` (the registered rule). The statistics and scoring code were exercised on synthetic
two-state data only (a code test, not a result). Executing the verification script on the tower was refused by the
session's permission check; the user decides how it is launched.

**09:00, dynamic verification done (run by the user on the tower; output pasted into the session) and batch launched.**
Configuration off the working point on purpose (G = 0.30, nsig = 0.040, dt = 0.01, seed 42); no statistic at the
working point has been seen.

- Installed files: the sha256 of `nb_mpr.py`, `nb-montbrio.py.mako`, `noise.py`, `integrators.py`, `infinite_theta.py`
  on the tower equal those of the copies read on 2026-11-24 (`sim/external/tvb_master/` for the first two).
- Model as configured by their `configure_sim`: η = −5, J = 15, Δ = 1, τ = 1; boundaries r ∈ [0, max float];
  `Linear` coupling a = G; `HeunStochastic`, dt = 0.01; nsig shape (2, 1, 1), values [0.04, 0.08].
- Connectome: N = 84, horizon 1245 steps; delays over weighted links 24 to 1244 steps; max weight 1.
- Rendered kernel: `r_int = bound_r(r_int)` after the predictor and `r_n = bound_r(r_n)` after the result; the same
  noise sample enters both stages (template lines 145 and 157); `bound_r` sets x < 0 to 0. Plain `@nb.njit`, one thread.
- Noise buffer sd per step: r 0.028289 (√(2·nsig·dt) = 0.028284), V 0.040004 (√(4·nsig·dt) = 0.040000). So this
  folder's σ (sd per √time unit) is √(2·nsig) on r and √(4·nsig) on V, as read from the source.
- Raw run, 20000 steps: finite; r is exactly 0 at 5.4% of (node, step) pairs and never below 0. The bound is active in
  the group's own scheme at this off-target point (the share at the working point is not known).
- Timing: 37 μs per step (84 nodes); their chunked path, 2000 time units in 7.2 s, output every 0.1 time units, finite.
  Estimate for one registered run at dt = 0.01: about 4 to 5 minutes; memory about 1.6 GB per process.
- Batch: `tvb_ref_batch.py 6` started detached on the tower at 09:00 (52 jobs; tower had 35.8 GB free and 4% load;
  the three Python processes already there are the user's own services, untouched).

## 2026-11-24 (10:00), stage A′ scored: ADEQUATE by the registered rule; two measures have not levelled off in dt
52 of 52 runs finite, exit 0 (`sim/runs/Aprime/A_*.json`, `runs_A_batch.log`; scorer `sim/tvb_ref_score.py`). Mean ± se
over seeds. Simulation: the companion pipeline unchanged on TVB 2.10.0; statistics: this folder's.

**Validity (registered): holds.** No non-finite run. Up-occupancy at the working point inside 0.05–0.95 in 7 of 8 seeds
(0.043 to 0.083; the state is mostly down with excursions, near the lower edge of "mixed"). Switching index 0.0142 ±
0.0014 at G = 0.54 against 0.0003 (G = 0.30) and 0.0004 (G = 0.80). Not registered, observed: VAF of two components of r
is 0.42 at the working point (the paper reports 0.41); the switching index over G = 0.53, 0.54, 0.55 is 0.0062, 0.0142,
0.0123 (0.54 against 0.55 is within 1 se).

| at G = 0.54, σ = 0.030 | dt 0.02 | dt 0.01 | dt 0.005 | dt 0.0025 (4 seeds) | G 0.53, dt 0.01 | G 0.55, dt 0.01 |
|---|---|---|---|---|---|---|
| m1 mean rate | 0.1585 ± 0.0036 | 0.1867 ± 0.0043 | 0.2054 ± 0.0063 | 0.2271 ± 0.0054 | 0.1492 ± 0.0025 | 0.2527 ± 0.0078 |
| m2 up-occupancy (r > 0.7) | 0.0399 ± 0.0032 | 0.0637 ± 0.0039 | 0.0795 ± 0.0058 | 0.0983 ± 0.0049 | 0.0298 ± 0.0023 | 0.1234 ± 0.0070 |
| m3 median up-fraction per TR | 0.0146 ± 0.0008 | 0.0231 ± 0.0023 | 0.0444 ± 0.0092 | 0.0744 ± 0.0134 | 0.0129 ± 0.0007 | 0.1074 ± 0.0143 |
| m4 5th percentile, same | 0.0019 ± 0.0001 | 0.0023 ± 0.0001 | 0.0031 ± 0.0002 | 0.0039 ± 0.0005 | 0.0017 ± 0.0001 | 0.0040 ± 0.0003 |
| m5 VAF, two components of r | 0.374 ± 0.010 | 0.423 ± 0.006 | 0.417 ± 0.005 | 0.410 ± 0.007 | 0.318 ± 0.012 | 0.409 ± 0.013 |
| switching index (secondary) | 0.0094 ± 0.0010 | 0.0142 ± 0.0014 | 0.0098 ± 0.0014 | 0.0122 ± 0.0032 | 0.0062 ± 0.0006 | 0.0123 ± 0.0019 |

**Rule (registered): ADEQUATE.** D_dt (dt 0.01 against 0.005) is smaller than D_G (one grid step) for all five primary
measures: ratios D_dt / D_G = 0.36 (m1), 0.34 (m2), 0.45 (m3), 0.76 (m4), 0.13 (m5). Four of the five differences are
beyond 2 se (2.5, 2.3, 2.2, 3.4 se), so the step does move the statistics; it moves them by less than one grid step.
The ⟦HAND⟧ size heuristic in the prior ("about half of one grid step") held for m1 to m3. Prior on ADEQUATE was 0.45.

**Trend (registered, scored separately).** m1 to m4 move monotonically across all four steps, always in the direction
of larger G (more up-state). Last increment (0.005 → 0.0025): m1 +0.0216 (2.6 se), m2 +0.0188 (2.5 se): NOT levelled
off. m3 +0.0300 (1.8 se), m4 +0.0007 (1.5 se): monotone, last increment within 2 se (4 seeds at the finest step). m5
not monotone, flat from dt 0.01 down. Prior on "still drifting at the finest pair" was 0.5.

**Post hoc arithmetic on the same numbers (not registered; description only).** Increments of m1 per halving: +0.028,
+0.019, +0.022; they do not visibly shrink (a √dt law would predict about +0.014 for the last, 1 se below what is seen;
the data do not separate the two). Total shift from their step to the finest tested (0.01 → 0.0025) against D_G: m1
0.78, m2 0.74, m3 1.08, m4 1.42, m5 0.27 of one grid step. Read through m1, refining the step from 0.01 to 0.0025 acts
on these statistics like about +0.008 in G. The dt → 0 values are not known; no extrapolation is recorded as a result.
Secondary: the switching index differs between dt 0.01 and 0.005 by 0.0043 ± 0.0020 (2.2 se; D_G = 0.0031) but is not
monotone across the four steps (0.0094, 0.0142, 0.0098, 0.0122); with these standard errors this is not evidence of a
step effect on the fitted statistic, nor against one.

**Per-TR up-fraction at dt 0.01 (registered description, quantiles only; histogram not made):** 5th percentile 0.002,
median 0.023, 95th percentile 0.226, sd 0.076. The network sits near all-down and makes partial excursions.

**What it means, under the stated assumptions (connectome, time convention, noise split, backend).** ADEQUATE is the
registered verdict on one halving; it is not "their step is fine". Both facts stand together:
- Under one halving of their step, four primary measures sit 2.2 to 3.9 se below one grid step. m4 is not separated
  from a grid step (D_G − D_dt = 0.0003 ± 0.0003): not shown to exceed it, no more.
- The level statistics (m1 to m4) drift in one direction at every halving from 0.02 to 0.0025 with no visible
  shrinkage (m2 rises 2.5-fold, m3 fivefold over that range); two halvings add up to about one grid step; the dt → 0
  values are unknown.
- Attribution, narrowed: a jump of the size seen earlier in this folder (frac_up 0.30 → 0.56 in one halving) is not
  seen in their pipeline at dt ≤ 0.02; a smaller drift of the same direction is. Their scheme was never run at dt 0.05,
  and stage A′ differs from stage A in connectome, G, σ, time convention and step range, so "the earlier effect belongs
  to this folder's rebuild" is NOT tested. The registered either/or ("their step is stable, or the fitted G depends on
  the step") does not fit the outcome: it is in between.
- VAF shows no drift below dt 0.01. The switching index (the nearest thing here to the paper's fitted statistic) is
  UNRESOLVED at this precision: D_dt 0.0043 ± 0.0020 against D_G 0.0031, not monotone in dt, and it sits at its peak in
  G where D_G is small by construction, so the yardstick is weak for it.
- The drift is not attributed to the bound at zero: the binding share at the working point was not measured.
- Sampling check: all 52 runs have 610000 samples at period 0.1; the delay horizon scales with 1/dt (623, 1245, 2490,
  4978 steps), so TR = 720 samples is the same window at every step.
- Discriminating follow-up (not started): where the switching-index peak sits in G at dt 0.005, which needs the G
  neighbours at that step and more seeds.
**Not done:** dt below 0.0025; the share of steps at which the bound binds at the working point; any (G, η) map; the
paper's own fluidity statistic (their BOLD model and window settings were not used). Horizon reached: step back.

## 2026-11-24 (10:20), the lab's own sweep code (`INN-SFU/tvb-node`, public, MIT; user: "the tvb repo we've been using for G Sigma investigation") and a zero-delay check, registered before running
**Read from the repository (commit 11f5215, 2026-06-08; workshop notebooks Module 2 and 3; the helper package `src`
is not in the repository, it ships in the container image `lrokos/tvb_node`).** Same TVB kernel as the INS companion
code: `NbMPRBackend().run_sim(sim, simulation_length)`, `HeunStochastic(dt = 0.01)`, `noise.Additive(nsig = [σ, 2σ],
noise_seed = 2)` (one fixed seed), `TemporalAverage(period = 0.1)`, `tavg_t *= 10`. Different from the INS settings:
η = −4.6, J = 14.5, Δ = 0.7, τ = 1; subject connectomes of 100 regions, weights used as loaded (no normalisation; mean
in-strength 1.13 and 1.15 for the two example subjects; zero diagonal; symmetric); **no delays**
(`tract_lengths = ones`, `speed = inf`); baseline G = 1.99, σ = 0.045; suggested sweep G 1.85–2.05, σ 0.035–0.050;
length 10e3 time units (100 s); BOLD by `simulation.tavg_to_bold(…, decimate = 2000)` (TR 2 s), first 8 samples
dropped; FCD with window 5; fit by maximum FCD variance, or KS distance to the empirical FCD, or FC correlation.
**Reading of the kernel at zero delay ⟦HAND, unverified⟧.** The template computes the corrector-stage coupling as
`cx(i, n, …)`, i.e. Σ_j w[n, j]·r[j, i − idelays[n, j]], inside the loop over nodes n at step i. With idelays = 0 this
reads `r[j, i]`: for j < n the value already updated in this step; for j ≥ n the pre-filled noise sample of node j
(the buffer holds the noise until it is overwritten). If so, the corrector sees coupling only from lower-numbered nodes,
the Heun average gives links from higher-numbered nodes half weight, and the result depends on the order of the nodes.
With delays of at least one step (the INS runs: 24 steps or more) the same line reads past states and is unaffected.
**Check (`sim/tvb_zero_delay_check.py`, tower, TVB 2.10.0).** Six-node random symmetric network and the lab's node
parameters; buffers captured at the kernel's entry; references written by hand in NumPy on the same buffers.
- P1: with the lab's connectivity construction, idelays are all 0 and the horizon is 1. Prior 0.97.
- P2: the default mode (`compatibility_mode = False`, what `run_sim` uses) differs from standard Heun (one coupling
  per step, from the previous step) and equals the reading above to round-off (max abs difference ≤ 1e-9 over 2000
  steps). Prior 0.9. **Defeats the reading:** default mode equal to standard Heun.
- P3: `compatibility_mode = True` equals standard Heun to round-off. Prior 0.9.
- P4: at σ = 0, relabelling the nodes in reverse order changes the default-mode trajectories (difference far above
  round-off after undoing the relabelling) and does not change the compatibility-mode ones. Prior 0.9.
- Size, reported without a prediction of magnitude: subject sub-0001 from the repository, baseline G = 1.99, σ = 0.045,
  10e3 time units, default against compatibility mode, same seed: mean rate, up-occupancy, per-node mean rate against
  node index. Direction prior: default mode has less effective coupling, so lower mean rate (0.7).

**Outcome (2026-11-24, 10:35; tower, tvb-library 2.10.0; `sim/tvb_zero_delay_check.py`, `sim/tvb_zero_delay_size.py`).**
One change to the notebooks' connectivity construction: `conn.compute_region_labels()` fails under numpy 2, so the
labels were set directly (labels do not enter the dynamics).
- P1 HELD as predicted: idelays are all 0, horizon 1.
- P2 HELD: over 2000 steps on the six-node network, the default mode differs from standard Heun by up to 3.8 in r and
  equals the reading to 9e-12 (r) and 3e-11 (V). The two agree exactly after 1 step and differ by 8e-3 after 10.
- P3 HELD: `compatibility_mode = True` equals standard Heun to 1e-13.
- P4 HELD: at σ = 0, reversing the node order changes the default-mode trajectories by up to 4.1 after undoing the
  relabelling; in compatibility mode the difference is 7e-15.
So in the default mode with zero delays the corrector stage of node n takes its coupling from the values already updated
in this step for nodes j < n and from the pre-filled noise samples for nodes j ≥ n. Observed through the real kernel
(default against compatibility mode; order dependence); the hand reference is this folder's.

**Size at the lab baseline (subject sub-0001 from the repository, σ = 0.045, 10e3 time units, first 1000 dropped; seeds
2 and 3; description, no registered prediction of magnitude; the direction prior "less effective coupling" held).**

| | G = 1.99: mean r, up-occupancy | G = 1.85: mean r, up-occupancy | Spearman(node mean r, node index) |
|---|---|---|---|
| default mode (as the notebooks run) | 0.456, 0.285 / 0.459, 0.288 | 0.214, 0.079 / 0.215, 0.080 | +0.64, +0.72 / +0.49, +0.56 |
| standard Heun, same W | 1.18, 0.86 / 1.18, 0.86, then non-finite | 1.14, 0.84 / 1.14, 0.84, then non-finite | −0.03, 0.00 / −0.06, 0.00 |
| standard Heun, W_eff (half weight from higher-numbered nodes) | 0.425, 0.259 / 0.421, 0.255 | 0.214, 0.079 / 0.219, 0.084 | +0.68, +0.64 / +0.55, +0.58 |
| default mode, node order reversed | 0.396, 0.235 / 0.443, 0.274 | 0.215, 0.081 / 0.216, 0.081 | +0.54, +0.45 / +0.33, +0.42 (against the reversed index) |

- Standard Heun at these G sits mostly in the high state and goes non-finite partway through every run (samples 41893,
  54530 at G = 1.99; 26870, 93321 at G = 1.85, of 100000). The default-mode runs stay finite.
- Standard Heun on W_eff (full weight from lower-numbered nodes, half from higher-numbered; mean in-strength 0.848
  against 1.131; node 0 keeps 0.50 of its input weight, the last node 1.00) reproduces the default-mode statistics
  closely (per-node mean rates: Spearman +0.98, +0.97, +0.96, +0.97), not exactly (0.425 against 0.456 at G = 1.99).
- Per-node mean rates under the original and the reversed node order are unrelated (Spearman −0.05, 0.00, 0.00,
  −0.05). Numbering alone would give a strongly negative value and the connectome alone a strongly positive one; about
  zero says the two contribute comparably to the pattern of activity over nodes at these settings.
**History of the line (tvb-root; both versions read).** The second coupling evaluation, indexing column i
(`r[m, i - idelays[n, m]]` under `% if not compatibility_mode`), was introduced in commit 63947ec68 "fixed heun
method" (2021-05-03); the first version (f68d6fd, 2021-03-31) computed the coupling once per step. The current
template has the same construction.
**Upstream check.** The lab's node parameters (η −4.6, J 14.5, Δ 0.7) are those of the INS ageing-brain code, which
runs with delays (`conn_speed = 2.0`); the Fousek paper states delays with speed 2 m/s (full text, lines 634–635);
the companion code uses speed 20. No infinite-speed construction was found in the saved upstream files. The
zero-delay construction (`tract_lengths = ones`, `speed = inf`) is in the lab notebooks' `get_connectivity`.
**Scope and what is NOT established.**
- Applies when the delay of a weighted link is 0 steps. The delayed runs made here (24 steps or more at dt 0.01, 12
  at dt 0.02) are unaffected, so stage A′ stands. Nothing is claimed about the paper's own runs beyond that it states
  finite delays.
- The lab's container (`lrokos/tvb_node`) was not run; its TVB version is not known. That it behaves the same is an
  inference from the history above and from the notebooks calling `run_sim` in its default mode.
- Whether the lab's research sweeps (as opposed to these workshop notebooks) run without delays is not known.
- The notebooks' fit statistic (FCD variance) was not computed: the helper package is not in the repository.
- One subject, two seeds, two values of G. No prior-art search yet on whether this is a known issue in TVB.

## 2026-11-24 (10:40), JAX port of the group's kernel: match and timing, registered before running
User: "lets do more fullsome parameter sweeps so we can get a nice 2d map for each partial derivative … use this as an
opportunity to optimize the implementation on the tower". The sweep itself is NOT designed or registered here: its
target (INS settings with delays, or the lab's settings) is the user's open decision. This entry covers the instrument.
**Kernel (`sim/mprjax.py`).** One step, as TVB's template with delays of at least one step on every weighted link:
c0 = G·Σ_j W[n, j]·r_j(i − 1 − D[n, j]); predictor x + dt·f(x, c0) + noise, r bounded at 0; c1 = G·Σ_j W[n, j]·r_j(i −
D[n, j]); result x + dt·(f(x, c0) + f(x_pred, c1))/2 + the same noise, r bounded at 0. Since c0 of step i + 1 equals c1
of step i when no weighted delay is 0, the coupling is gathered once per step and carried (asserted, not assumed).
History of r in a ring buffer; parameters (G, η, J, Δ, σ) per batch row; noise shared across rows of the same seed.
**M1, exact match (laptop CPU, float64).** TVB's own buffers (initial history and pre-filled noise, captured at the
kernel's entry: `sim/tvb_ref_dump.py`, G = 0.54, σ = 0.030, dt = 0.01, 3000 steps) are fed to the JAX step. Pass:
max |r_JAX − r_TVB| and max |V_JAX − V_TVB| ≤ 1e-8 over the 3000 steps, with the growth reported. Prior 0.85 (loss:
round-off amplified by the dynamics, or a misread index). **Defeat:** a difference above 1e-6 in the first 100 steps
means the port is not their kernel; nothing else is run on it until found.
**M2, statistics on the GPU (float32, own random numbers and initial state).** At (G, σ) = (0.54, 0.030), seeds 1–8,
61000 time units with the first 1000 dropped, dt = 0.01 and 0.005: the five primary measures of stage A′ against the
TVB values recorded at 10:00. Pass: every one of the ten comparisons within 3 combined se, and at most two beyond 2 se.
Prior 0.75. A failure is looked for in precision (float32), initial state and noise scaling, in that order.
**Timing (description, no prediction scored).** Steps per second and seconds per run for batch sizes 8, 64, 256, 1024
(and larger if memory allows) on each GPU, float32; against TVB's numba kernel on the same machine (measured: 37 μs
per step per run on one core; 3.9 minutes per run with six in parallel).
**Horizon.** M1, M2, timing. No sweep before the user picks the target and the design is registered.

**M1 outcome (2026-11-24, 10:55; laptop CPU, float64; `sim/mprjax_match.py` on `sim/runs/port/tvb_ref_dump.npz`).**
The JAX step, fed TVB's captured initial history and noise, reproduces TVB's trajectory: max |difference| 2e-16 over the
first 10 steps, 2e-15 over 100, 3e-13 over 1000, 4e-12 over all 3000 (criterion 1e-8). The share of values exactly at
zero is the same in both (2.14%). Delays computed as rint(L / speed / dt) equal TVB's idelays at all four steps. So the
carried-coupling ring-buffer step is their kernel, for this connectome and these delays. The batched runner
(`sim/mprjax_run.py`: on-device noise, temporal average, online statistics) ran end to end on the laptop on a short
job (a plumbing test; its statistics are validated only by M2).

**M2 outcome (2026-11-24, 11:25; tower GPU 0, float32; `sim/mprjax_m2_score.py` on `sim/runs/port/m2_dt*.json` against
`sim/runs/Aprime/`).** All 48 JAX runs finite. Registered comparisons at (G, σ) = (0.54, 0.030), JAX − TVB, in combined
se: dt 0.01: mean rate −0.011 (−2.0), up-occupancy −0.010 (−2.0), median up-fraction −1.1, 5th percentile −0.8, VAF
−1.7; dt 0.005: −0.6, −0.6, −1.2, −1.0, +1.3. Two of ten beyond 2 se (both at 2.0, and the two measures move together),
none beyond 3: within the registered criterion, at its edge. Description, not registered: at G = 0.53, 0.55, 0.80 every
measure is within 0.6 se; at G = 0.30 mean rate differs by +0.0002 (0.17%; +2.6 se, TVB n = 4) and up-occupancy by
+2.1 se. Secondary BOLD measures within 1.4 se everywhere.
**Follow-up registered before running (precision).** The working-point differences are all negative at dt 0.01. To
separate chance from single precision: JAX only, (0.54, 0.030), dt 0.01, 32 seeds in float32 and 32 in float64 (the
noise draws differ between the two, so this is a comparison of means). Concern if any of the five primary measures
differs by more than 3 combined se between the two precisions; then production runs use float64 or the cause is found.
Prior that they agree within 3 se: 0.85.

**Precision follow-up, outcome (2026-11-24, 11:35; 32 seeds each, (0.54, 0.030), dt 0.01; `sim/runs/port/prec_f32.json`,
`prec_f64.json`).** All finite. float32 − float64: mean rate +0.0019 (+0.5 se), up-occupancy +0.5 se, median
up-fraction +0.9 se, 5th percentile −0.1 se, VAF −0.1 se: agree, no concern by the registered criterion. Secondary:
switching index 0.0102 ± 0.0007 against 0.0127 ± 0.0009 (−2.2 se; not in the criterion; noted). With 32 seeds the JAX
mean rate is 0.180 (float32) and 0.178 (float64) against TVB's 0.187 ± 0.004 from 8 seeds (−1.3 and −1.7 se): the port
sits slightly below the TVB estimate in both precisions, inside 2 se. More TVB seeds at the working point would
tighten this; not run.

**Timing (description).** Same machine, 61000 time units at dt 0.01 (6.1 million steps), 84 regions.
- TVB numba kernel: 37 μs per step per run on one core; 3.9 minutes per run, six at a time, so 39 s of wall time per run.
- JAX, first version (history updated every step with an indexed assignment): 3.6 μs per step per run at any batch size
  on GPU 0, about 22 s per run. The cost did not depend on how the delayed values were gathered: the whole history
  buffer was being copied at every step (same signature on the laptop CPU).
- JAX, current version (`sim/mprjax.py`: batch as the last axis, nonzero links only, history constant inside a chunk of
  one temporal-average sample and updated once per chunk with a slab; valid because every weighted delay is at least
  one chunk long, asserted): 0.40, 0.21, 0.158, 0.153 μs per step per run at batch 64, 256, 1024, 2048 on GPU 0 (RTX
  3060), including noise generation and all online statistics; 0.93 s per run at batch 2048. About 40 times less wall
  time per run than the six-process TVB batch. float64 costs about twice float32 at batch 32. Batch 4096 does not fit
  the 8 GB card; not tried on the 12 GB card with the current version.
- Both versions pass M1 (current version: 2e-12 over 3000 steps). First version kept as `sim/mprjax_v1_perstep.py`.
**State.** The instrument is matched and timed for the INS settings with delays. No sweep is designed or run: the
target is the user's decision (asked 10:35). The kernel does NOT cover zero delays (it asserts against them); the
lab-notebook setting would need the as-run corrector rule added and matched separately.

## 2026-11-24 (12:05), derivative maps on the INS settings: pilot and sweep, registered before running
User: "go" (target: INS settings with delays; companion pipeline's connectome, speed 20, [σ, 2σ], dt 0.01; kernel
`sim/mprjax.py`, float32, matched at M1/M2).
**Checked before designing (existing runs, `m2_dt0.01.json`).** Sharing the noise across parameter values does NOT
reduce the noise of differences here: for G 0.53 → 0.55 on the same eight seeds the se of the paired difference in mean
rate is 0.0080 against 0.0068 unpaired (correlation across seeds −0.8, 0.0, −0.1 for the three pairs). So differences
carry the full seed-to-seed scatter (sd of mean rate per run about 0.012 at the working point), and offsets in η, J, Δ
must produce effects of a sizeable fraction of one G notch to be seen. Host-side BOLD statistics: 0.055 s per run.
**Forced and expected structure, written before any map ⟦HAND, unverified⟧.**
- Scaling. With τ = 1 the node is ṙ = Δ/π + 2rV, V̇ = V² + η + Jr − π²r² + G·c. Put r = √Δ·r̃, V = √Δ·ṽ, t = t̃/√Δ:
  the equations keep their form with Δ → 1, η → η/Δ, J → J/√Δ, G → G/√Δ, and the noise level σ → σ·Δ^(−3/2); delays
  and every time window scale by √Δ, rates by √Δ. So Δ is not an independent parameter of the model: to the extent a
  statistic does not care about the time scale, the delay scale or a fixed threshold, its derivatives satisfy
  Δ·m_Δ + η·m_η + ½J·m_J + ½G·m_G + (3/2)σ·m_σ = k·m, with k = ½ for a rate and 0 for a dimensionless statistic. A
  near-linear dependence of the Δ column on the others is therefore forced, not a finding; the residual of this
  identity measures how much the fixed delays, windows and thresholds break it. (The same nondimensionalisation is, as
  far as I recall, in Montbrió, Pazó and Roxin 2015; not rechecked today.)
- η, J·r and G·c enter the V equation as a sum, so the η, J and G columns are expected to be close to proportional
  wherever the state is homogeneous. Prior that the η and G columns have cosine above 0.95 over most of the mixed
  region: 0.8. **Defeats "G is confounded with η on these statistics":** cosine below 0.8 over most of that region.
- σ is expected to be the best-separated direction (prior 0.6).
**Pilot (one batch, also the first full-length batch at production size).** Points (G, σ) = (0.45, 0.030),
(0.54, 0.030), (0.65, 0.030); at each: base, η ± {0.005, 0.01, 0.02}, J ± {0.02, 0.05, 0.10}, Δ ± {0.005, 0.01, 0.02};
16 seeds; 57 configurations, 912 runs, 61000 time units, first 1000 dropped.
- Offset rule, fixed now: for each parameter take the smallest δ whose central difference in mean rate at the working
  point, m(+δ) − m(−δ), is at least 3 se at the sweep's seed count; signal comes before symmetry (a central difference
  cancels the even part), but a δ whose one-sided response exceeds 0.10 in mean rate (two G notches) is too large and is
  not taken. The asymmetry |Δ₊ + Δ₋| / (|Δ₊| + |Δ₋|) is recorded for the chosen δ.
- Seed rule, fixed now: 4 seeds per configuration if every chosen δ reaches 3 se with 4; otherwise 8.
- Timing and memory of this batch set the sweep's batch size and the duration quoted to the user.
**Sweep (launched only after the pilot fixes δ and seeds; its numbers are appended here before launch).** Grid G =
0.40, 0.41, …, 0.70 (31) × σ = 0.020, 0.024, …, 0.056 (10); at each point base and η ± δ_η, J ± δ_J, Δ ± δ_Δ (7
configurations) at dt 0.01, and the base again at dt 0.005. Controls: the σ = 0.030 row, base only, in float64; 24 more
TVB seeds at the working point on the tower CPU.
**Analysis, fixed now (`sim/sweep_maps.py`, to be written while it runs).**
- Statistics: mean rate, up-occupancy, median up-fraction per TR, VAF of r (two components), switching index, mean FC,
  corr(FC, SC). Derivatives: G and σ by central differences on the grid (one-sided at the edges); η, J, Δ by central
  differences of the offsets; step size as the difference dt 0.005 − dt 0.01 (one halving).
- A derivative is shown only where it exceeds 2 se (seed scatter, unpaired); elsewhere the cell is blank. Non-finite
  cells are holes.
- For the peaked statistics (switching index, VAF) the reported quantity is the location in G of the row's maximum
  (after a three-point smooth) for the base and for each variant: its shift per variant says whether the fitted G moves.
- Local confounding: at each grid point where the G column has norm above 3 se, the cosine between the G column and each
  other column over the seven statistics (each statistic scaled by its seed sd at that point), and the ratio of
  derivatives ("equivalent G shift per unit of the other parameter") with its spread across statistics.
- "Mixed region" for the priors above: grid points with base up-occupancy between 0.02 and 0.90.
**Horizon.** Pilot, sweep, maps, then step back. No further variants or regions before that.

**Pilot outcome (2026-11-24, 14:50; one batch of 912 runs on GPU 0, 16 min, 1.05 s per run, all finite;
`sim/runs/sweep/pilot_done.json`, `sim/sweep_pilot_score.py`).** Base at the working point, 16 seeds: mean rate 0.187
(sd per run 0.012), switching index 0.0132 (sd 0.0042), VAF 0.416 (sd 0.013).
- Offsets by the registered rule (mean rate at the working point): δ_η = 0.005 (central difference +0.048, 5.1 se at 4
  seeds, asymmetry 0.05), δ_J = 0.02 (+0.080, 8.2 se, asymmetry 0.17), δ_Δ = 0.005 (+0.052, 5.7 se, asymmetry 0.14).
  Seeds per configuration by the registered rule: 4.
- Size, in plain terms: at the working point raising η by 0.005 (0.1% of its value) raises the mean rate by 0.023,
  about what one notch of G (0.01) does (+0.025 from 0.53 to 0.54); J + 0.02 (0.13%) gives +0.047; Δ + 0.005 (0.5%)
  gives +0.023. At G = 0.45 and 0.65 the same offsets move the mean rate by 0.001 and 0.005–0.013.
- The pattern statistics are NOT resolved at these offsets with 4 seeds (switching index 0.4–1.2 se, VAF 2.3–3.4 se
  at the working point); they are analysed by peak location, as registered. Seeds 5–8 can be added later as further
  batches without wasting these.
**Sweep numbers, fixed before launch.** δ_η = 0.005, δ_J = 0.02, δ_Δ = 0.005; 4 seeds; σ grid shifted to 0.018,
0.022, …, 0.054 so that it contains the paper's 0.030 (the 12:05 text said 0.020 … 0.056); float64 control row at σ =
0.030. 2170 configurations at dt 0.01, 310 at dt 0.005, 31 in float64: 10044 runs in 12 batches of up to 1024.
**TVB with 32 seeds at the working point (control, done).** TVB mean rate 0.1822 ± 0.0022 (the first 8 seeds gave
0.187). Port against TVB, 32 seeds each: float32 mean rate −0.6 se, up-occupancy −0.6, median up-fraction +0.8, VAF
−1.8; float64 −1.2, −1.2, −0.3, −1.7. The level statistics agree. Switching index: TVB 0.0129 ± 0.0006; port float64
0.0127 ± 0.0009; port float32 0.0102 ± 0.0007 in one batch (seeds 1–32) and 0.0132 ± 0.0010 in the pilot batch (seeds
1–16, same seeds, different batch: GPU arithmetic order differs, so these are different realisations).
**Precision check on the switching index, registered before running.** 64 new seeds (101–164) at the working point in
float32 and in float64. If the float32 mean differs from the float64 mean by more than 3 combined se, the sweep runs in
float64 (about twice the time); otherwise in float32 with the float64 row as control. Prior that it stays within 3 se:
0.7.

**Precision check on the switching index, outcome (2026-11-24, 14:58; 64 seeds each, `sim/runs/port/prec2_f32.json`,
`prec2_f64.json`).** float32 0.0111 ± 0.0006 against float64 0.0114 ± 0.0006 (−0.4 se); mean rate +1.2 se, VAF 0.0 se,
mean FC +1.0 se. Within the registered criterion: the sweep runs in float32.
**Open item (not a precision effect).** Pooling every working-point run so far: mean rate TVB 0.1822 ± 0.0022 (n = 32),
port 0.1791 ± 0.0014 (float64, n = 96) and 0.1826 ± 0.0014 (float32, n = 112): agree. Pattern statistics sit lower in
the port than in TVB at about 2 se each (they move together, so this is one signal, not three): switching index 0.0129
± 0.0006 (TVB) against 0.0119 ± 0.0005 and 0.0112 ± 0.0004; VAF 0.415 ± 0.004 against 0.406 ± 0.002 and 0.407 ± 0.002;
mean FC 0.189 ± 0.004 against 0.175 ± 0.003 and 0.179 ± 0.003. Unexplained. The two statistics paths (offline on TVB's
output; online in the port) were not compared on identical input. Derivative maps are internal to the port, so they
do not depend on this; comparisons of pattern-statistic LEVELS with TVB do.
**Sweep launched 14:59** (tower, two detached containers `sfm_w0`, `sfm_w1`, one per GPU, queue `q_sweep`; 12 batches;
float32; BOLD kept on the tower in the batch `.npz` files).

**Statistics paths compared on identical input (2026-11-24, 15:10; laptop CPU, float64; `sim/stats_path_check.py`).**
Two runs at the working point, 140 TRs, every temporal-average sample kept: the port's online statistics and the
offline statistics used for the TVB runs agree to machine precision on the same rates (BOLD identical; mean rate,
up-occupancy, quantiles, VAF, mean FC, corr(FC, SC) equal to 1e-10 or better; the record is too short for the switching
index, which is computed from that identical BOLD). So the lower pattern statistics in the port (about 2 se) are not
explained by the step (M1), by precision, or by the statistics code. Remaining: chance, or something in TVB's long runs
that a 3000-step match does not cover (initial history, chunking). 64 more TVB seeds (33–96) launched on the tower CPU
at 15:10 to tighten the comparison. User, on my first wording: "two standard errors is more than just 'slightly lower'".

**Burn-in check (2026-11-24, 15:18; user: "chain burn in?"; `sim/burnin_check.py` on the per-TR arrays saved on the
tower; working point; TVB n = 32, port n = 80 (32 float32, 32 float64, 16 pilot); description, not registered).**
- A transient outlasts the 1000-unit discard in BOTH codes. Mean up-fraction per TR over seeds, at TR 1, 5, 10, 20,
  50, 100 after the discard: TVB 0.158, 0.118, 0.106, 0.080, 0.041, 0.053; port 0.130, 0.106, 0.097, 0.063, 0.066,
  0.062; long-run value about 0.057. First 20 TRs: 0.111 ± 0.011 (TVB), 0.093 ± 0.007 (port). So the runs are still
  relaxing for roughly 20 to 50 TRs (1400 to 3600 time units) after a discard of 1000; both start from rates uniform on
  [0, 2]. The BOLD statistics drop a further 41 TRs and so miss most of it; the rate statistics include it (first
  quarter of the record 0.062–0.066 against 0.055–0.058 later).
- It does NOT explain the gap between TVB and the port in the pattern statistics. Switching index on the first and
  second half of each record: TVB 0.0099 ± 0.0011 and 0.0123 ± 0.0011; port 0.0083 ± 0.0007 and 0.0084 ± 0.0005. The
  gap is in both halves and is larger in the second (3.2 se there; 1.2 se in the first). Mean FC by quarter: TVB 0.146,
  0.159, 0.179, 0.166; port 0.151, 0.153, 0.155, 0.154. Caution: this is the same data cut several ways after the fact.
- TVB's initial history differs between seeds (`sim/tvb_init_check.py`) and has the distribution the port uses, so a
  shared starting state is not the cause either.
- Still open. Eliminated so far: the step (M1), precision, the statistics code, burn-in, shared initial state.
  Pending: 64 more TVB seeds (running).

## 2026-11-24 (17:35), sweep scored: derivative maps on the INS settings; the TVB–port gap did not hold up
Registered at 12:05 (design, analysis rules, priors). Inputs: `sim/runs/sweep/done/*.json` (12 batches, copied from the
tower), `sim/runs/Aprime/` (TVB, now 96 seeds at the working point). Scripts: `sim/sweep_maps.py` (registered
analysis), `sim/sweep_post.py`, `sweep_post2.py`, `sweep_post3.py` (description and controls; post hoc items are
labelled as such below), `sim/burnin_check2.py`, `sim/sweep_nonfinite_check.py`. Figures:
`figures/` (moved out of `notes/` on 2026-11-24 when the folder became a repository, then sorted into `maps/`, `derivatives/`, `tables/`, `report/`; `base.png`, `d_<parameter>.png`, `confounding.png`, `scaling.png`, `peaks.txt`,
`summary.json`). Everything here is the JAX port in float32 at dt 0.01 unless stated; assumptions as at 08:00
(connectome, time convention, noise split).

**The open item of 14:58/15:18 is closed: chance, not a difference between the codes.** The 64 new TVB seeds (33–96)
give switching index 0.0112 ± 0.0005, VAF 0.4035 ± 0.0036, mean FC 0.172 ± 0.004, mean rate 0.1789 ± 0.0019. The port
(224 runs pooled): switching index 0.0111 ± 0.0004 (float32) and 0.0122 ± 0.0005 (float64), VAF 0.407 and 0.406, mean
FC 0.179 and 0.177, mean rate 0.182 and 0.179. All 96 TVB seeds against all port runs: within 0.4 se on six
statistics, 1.5 se on the median up-fraction. The first 32 TVB seeds ran high (switching 0.0129, mean FC 0.189): about
2 to 3 se above TVB's own next 64. By halves of the record, TVB seeds 33–96: 0.0078 and 0.0087 (port 0.0083 and
0.0084). The "3.2 se in the second half" at 15:18 was the first 32 seeds cut after the fact; my reading then that the
split made the gap "look more real" was wrong. Left over, inside the port: the float32 and float64 pools differ by 1.7
se on the switching index (0.0111 against 0.0122, TVB's 0.0117 between them); the registered 64-seed check gave 0.0111
against 0.0114. Not pursued; it does not enter the maps, which are float32 throughout.

**Runs.** 10044 runs of 61000 time units in 12 batches; 14400 s of GPU time over the two cards (about 1.0 s per run on
the RTX 3060, 1.4 on the RTX 3050; dt 0.005 about 2.8; float64 5.4). The same runs at TVB's measured 39 s per run on
this machine would be about 109 hours.

**Non-finite runs (deviation from the registered rule, corrected).** 44 runs (0.44%) ended non-finite: all at σ ≥ 0.038
and G ≥ 0.55, 24 of them at σ = 0.054 (there, for G ≥ 0.63, about one run in ten); 41 at dt 0.01 and 3 at dt 0.005,
spread over all seven variants. In the batch inspected (24 of the 44) the rates themselves become non-finite, in all
nodes, at a time anywhere between TR 28 and TR 787. The float64 row (σ 0.030) does not reach that region and TVB was
not run there, so whether this belongs to the scheme or to float32 is NOT established. `sweep_maps.py` kept the means
of the surviving seeds; the registered rule says such cells are holes. With the 26 affected grid points removed the
summaries below barely move (η: share above 0.95 goes 0.84 → 0.83). `sweep_maps.py` was then changed (`MIN_SEEDS = 4`:
a cell with fewer than 4 finite seeds is a hole) and the figures and `summary.json` regenerated (17:40). Numbers below
are with holes removed, except the peak locations (`peaks.txt` uses whatever seeds a cell has).

**Registered priors.** Mixed region as registered (base up-occupancy 0.02–0.90): 249 of 310 points; the upper bound
never binds, so this is everything except the quiet corner (low G, low σ). 218 points after holes and the 3 se rule.
- G against η (prior 0.8 that the cosine exceeds 0.95 over most of the region; defeated if below 0.8 over most):
  median cosine +0.985; above 0.95 at 83% of points; above 0.8 at 98%. As predicted.
- G against J: median +0.992; above 0.95 at 89%. G against Δ: +0.989; 84% (forced by the scaling, see below).
- σ best separated (prior 0.6): yes. |cosine| above 0.95 at 28%. Its sign flips across the transition: for G ≤ 0.55
  median +0.78 (more noise acts like more coupling), for G ≥ 0.57 median −0.94 (more noise acts like less coupling);
  up-occupancy is nearly independent of σ along G ≈ 0.56 (0.205–0.213 for σ 0.034–0.054, over TRs 51–833). The signed median over the
  whole region (−0.66) and a single "G-equivalent of σ" mean nothing.
- One halving of the step against G: median cosine +0.90; above 0.95 at 30%.
**Caveats on those cosines.** (i) Three of the seven statistics are one statistic: mean rate, up-occupancy and median
up-fraction correlate at 0.997–0.999 across the region, so the registered cosine is high cheaply. POST HOC, without the
two duplicates: η 0.971 (67% above 0.95), J 0.986 (80%), Δ 0.978 (71%); pattern statistics alone (VAF, switching
index, mean FC, corr(FC, SC)), 140 points: η 0.958 (55%), J 0.978 (76%), Δ 0.965 (59%). (ii) Noise pulls a cosine down.
POST HOC ceiling on the pattern statistics: each column estimated from seeds 1–2 against the SAME column from seeds
3–4 gives median cosine 0.976 for G, 0.955 for J, 0.891 for Δ, 0.861 for η (a column agrees with itself only this
well at two seeds). Across columns and across the same split: G against J 0.958, G against Δ 0.938, G against η 0.922.
The geometric means of the self-agreements are 0.965, 0.933, 0.917. So G's column agrees with the η, J and Δ columns
as well as those columns agree with themselves: the shortfall from 1 is noise. No separation of η, J or Δ from G is
shown anywhere, and none is excluded below the noise. (Seven statistics, same split: self 0.993, 0.992, 0.984, 0.978;
cross 0.986, 0.982, 0.976.)

**Sizes (registered: equivalent shift in G per unit of the other parameter; quartiles over the region).**
- η: 0.77 / 0.86 / 1.01. J: 0.38 / 0.41 / 0.44. Δ: 0.95 / 1.02 / 1.14. One halving of dt: 0.0020 / 0.0036 / 0.0058.
- In words, median: one G notch (0.01) does what η + 0.012 (0.23% of η) does, or J + 0.025 (0.16%), or Δ + 0.010
  (1%), or between two and three halvings of the step. Below the transition (G ≤ 0.55) the equivalents are larger
  (η 1.04, J 0.46, Δ 1.18, dt 0.0062); above it smaller (0.78, 0.39, 0.99, 0.0027).
- Per statistic (registered "spread across statistics"), median ratio where both derivatives exceed 2 se. η: mean
  rate 0.84 (221 points), VAF 0.91 (137), mean FC 1.07 (50), corr(FC, SC) 0.99 (32), switching index 1.8 (17). J: 0.41,
  0.38, 0.34, 0.38, 0.44. Δ: 1.00, 1.04, 1.12, 1.04, 1.6. dt: 0.0049, 0.0071, 0.0079 (16 points), then too few. The medians
  agree to about 10–20% for VAF, mean FC and corr(FC, SC). The point-by-point test is not clean: VAF's ratio is within
  2 se of the mean rate's at 72–86% of shared points and mean FC's at 82–96% (95% expected if the se were right and
  the ratios equal), and keeping only points above 2 se inflates the ratios of the noisy statistics (mean FC 1.07,
  switching index 1.8). The switching index resolves at 13–26 points only.
- ⟦HAND, unverified⟧ reading: η, J·r and G·c enter V̇ as a sum, so one unit of η is worth 1/c of G and one unit of J is
  worth r/c; the medians correspond to an effective c ≈ 1.2 and r ≈ 0.5. Not checked against the simulated c and r.
**Working point (0.54, 0.030), 4 seeds: it sits on a knee.** Mean rate 0.153, 0.182, 0.258 at G 0.53, 0.54, 0.55: the
notch below is +0.029, the notch above +0.076; the central slope (5.2 per unit G) averages them. η + 0.005 gives
+0.027 ± 0.008, η − 0.005 gives −0.022 ± 0.006; J ± 0.02: +0.061, −0.028; Δ ± 0.005: +0.043, −0.029. So the statement
made after the pilot, "η + 0.005 is about one G notch", holds against the notch below (0.53 → 0.54) only; against the
notch above it is about a third, and by central slopes about a half (0.0040–0.0047 in G).

**Peak locations (registered).** Location in G of the row maximum, mean over the ten σ rows of peak(+δ) − peak(−δ),
se over rows, against 2δ × the median equivalent with the sign reversed. The grid step is 0.01, so single rows are
quantised (entries "± 0.000" in `peaks.txt` are that, not precision).
- VAF (base peak 0.52–0.55 by row): η −0.010 (expected −0.008); J −0.014 ± 0.002 (−0.016); Δ −0.008 ± 0.001 (−0.010);
  dt halved −0.005 ± 0.002 (−0.004).
- Mean FC (POST HOC, not among the registered peaked statistics; base 0.54–0.55): η −0.010 ± 0.002; J −0.014 ± 0.003; Δ −0.011 ± 0.002; dt halved −0.003 ± 0.002.
- Switching index (base 0.54–0.57): η −0.009 ± 0.003; J −0.007 ± 0.005; Δ −0.005 ± 0.005; dt halved −0.001 ± 0.004.
  Resolved for η only. At 4 seeds the working-point cell alone reads 0.0085 (float32) and 0.020 (float64).
So the G at which the peaked statistics peak moves with η, J, Δ and the step by the amounts the level statistics
predict: a "best G" read off a peak is a position along the same combination.

**Scaling identity (forced structure).** The Δ column predicted from the other four by the hand identity of 12:05
against the measured one, quartiles of measured/predicted over 223 points: mean rate 0.84 / 0.99 / 1.22 (correlation
over points 0.82; median residual 18% of the Δ term); up-occupancy 0.86 / 1.05 / 1.33. For VAF (0.05 / 0.57 / 0.94)
and the other pattern statistics (medians 0.06–0.15) the prediction is a sum of four noisy terms each multiplied by 5
to 7.5, two of which nearly cancel, so those ratios are attenuated by noise: untested there. The residual in se
units has median 0.33, half of what correctly scaled independent errors would give (0.67), so the se used is too wide;
the z map (`scaling.png`) is not to be read as a test. For the level statistics the Δ map is, in the median, the one
the others force (spread of about ±20% point by point, not separated from noise); it is not a separate finding.

**Step size across the grid (dt 0.005 against 0.01).** Along σ = 0.030 the mean rate rises at every G when the step is
halved: +0.002 to +0.003 below G 0.51 (2%), +0.014 ± 0.004 at the working point, +0.034 ± 0.017 at 0.55, +0.007 at
0.60–0.62 (1%). As a shift in G: median 0.0036 per halving (0.0062 below the transition), in line with stage A′ in
TVB ("one halving ≈ a third of a notch").

**Controls.**
- float64 row (σ 0.030, base, 4 seeds, same seeds along the row so the 31 cells are not independent checks). Mean
  rate: differences of at most 0.0002 for G ≤ 0.47; within noise through the transition; for G 0.61–0.70 float64 is
  higher at all ten points by 0.0002 to 0.005 (under 1%). Pattern statistics at G 0.51–0.54: float64 higher (switching
  index +0.002 to +0.012; mean FC +0.05 ± 0.011 at G 0.52), while the registered 64-seed check at the working point
  showed no difference (0.0111 against 0.0114). Not read either way; not chased.
- Burn-in across the grid (`burnin_check2.py`, base cells; up-occupancy over all 833 TRs against TRs 51–833). At the
  working point the first 50 TRs add 0.0015 to 0.058 (under 3%). At low noise just below the transition they matter:
  at σ 0.018, G 0.46–0.54 the up-fraction over the first 20 TRs is 0.05–0.36 where the later value is 0.0004–0.21, and
  the whole-record up-occupancy is 0.026 against 0.013 without the first 50 TRs at G 0.52. 10 of 310 points differ by
  more than 0.005, all at σ ≤ 0.026 and G 0.50–0.54 (at σ 0.026, G 0.54: 0.035 against 0.029). So "the sweep is
  unaffected by burn-in" (said at 15:18) holds at and above σ 0.030 and NOT for the three lowest rows (σ 0.018,
  0.022, 0.026) near the transition: there the level statistics, and probably
  the VAF band at σ 0.018 from G 0.43 to 0.56 in `base.png` (not tested directly), carry the decay from the starting
  state (rates uniform on [0, 2]), as they would in the group's pipeline with the same discard.

**What is and is not shown.** Shown, for this connectome, these seven statistics, this pipeline and float32: over the
transition region the statistics respond to G, η, J and Δ along one direction to within the noise of 4 seeds, with
the exchange rates above; σ is seen separately, with opposite sign on the two sides. Weaker, for the integration step:
halving it moves the level statistics and the VAF and mean-FC peaks in the direction of more G (about 0.004 in G),
but its column is the least resolved (median cosine 0.90; VAF's exchange rate 0.0071 against the mean rate's 0.0049),
so "the step acts along the same direction" is supported for the level statistics and open for the rest. NOT shown:
that no statistic separates G from η (per-node quantities were saved and not analysed; the paper's own fluidity
statistic is not among the seven); anything for the switching index beyond the η peak shift; the three lowest σ rows as
stationary statistics; the high-G, high-σ corner; any of it in TVB itself (the port is matched step for step over 3000
steps and in distribution at one point). The additive entry of η, J·r and G·c makes the result expected (prior 0.8);
what the sweep adds is the sizes, that the peaks of VAF and mean FC move by the amounts those sizes predict, and
that the step shifts the level statistics the same way.
**Horizon reached** (pilot, sweep, maps): stepping back. No further variants or regions were run.

**Write-up of the sweep (2026-11-24, 17:51; user: "a succinct write up with figures and distributions of chains with SE's").**
`reports/sweep_writeup.pdf` (first written as an HTML page, a PDF built with LaTeX since 18:35 the same day; built by `sim/writeup_sweep.py`, which computes every
number in it from the run files) with five figures `figures/report/w1…w5_*.png`: per-run distributions
at the working point (TVB 96, port float32 124, port float64 100); per-volume traces after the discard (TVB 96, port
64; `sim/runs/port/traces_wp.npz`, extracted on the tower by `sim/dump_traces.py`); the σ 0.030 row along G; the η, J,
Δ and step offsets at the working point (pilot, 16 runs per value; TVB at four steps); the maps and cosines with holes.
Machine-written draft; not rewritten in the user's words; no new runs and no new claims beyond the 17:35 entry.

## 2026-11-24 (21:22), two checks registered before running: per-region separation of G from η; what the grey cells are
User: "formalize the hand elements then run the computations" (per-region analysis) and "a more exact analysis as to
whats going on with those grey cells". Nothing below has been looked at: the per-region arrays are still on the tower.

### A. Do per-region mean rates separate G from η?
**Formal statement (replaces the hand remark made in the session).** Write the noise-free network as ẋ = F(x; θ),
x = (r, V). At a steady state F = 0, so a small change δθ moves it by δx = −(D_x F)⁻¹ (∂F/∂θ) δθ. The parameter enters
only the V equations: ∂F_V,i/∂η = 1, ∂F_V,i/∂G = c_i with c_i = Σ_j W_ij r_j (the region's coupling input), and
∂F_V,i/∂J = r_i. So the response to G equals a multiple of the response to η for every region if and only if c is
constant across regions. Here in-strength runs from 0.20 to 4.90 (10th to 90th percentile 0.37 to 2.87), so c is far
from constant: at a steady state the two responses differ by construction.
- This is already checked numerically: the entry of 2026-11-23 (night), T4 and T5, found G and η structurally
  identifiable from a record of rates with the other parameters fixed (76-region connectome, undelayed, noise-free;
  cosine between the G and η columns 0.87–0.88). **So the deterministic case is forced, not a finding.**
- Smallest case, to be run through `src/core/mprjax.py` at σ = 0: three regions in a chain (in-strengths 1, 2, 1).
  Prediction: the middle region's share of the response is larger under G than under η. Prior 0.9.
- **The open question** is the switching regime at 4 seeds: does the time-averaged mean rate of each region carry the
  difference above the noise? Scope: this connectome, the sweep's runs, statistic = per-region mean rate
  (`node_mean_r`); per-region up-occupancy as description only.
**Statistic (one, directed, fixed now).** At a grid point, a_i = response of region i's mean rate to G (central
difference over the two neighbouring G values), b_i = response to η (central difference over η ± 0.005). Remove from
a its best multiple of b: e = a − κ b with κ = (a·b)/(b·b). T = Pearson correlation across the 84 regions between
e_i and the in-strength s_i = Σ_j W_ij. Prediction: T > 0 (G favours strongly connected regions beyond what η does).
- Replication unit: the seed. T is computed per seed (regions switch together, so 84 regions are not 84 samples),
  averaged over grid points; the four per-seed values give mean and se.
- Grid points used: mixed region (base up-occupancy 0.02–0.90), without holes, without the rows σ ≤ 0.026 (burn-in),
  and with both neighbours in G present.
- **Noise floor first, before η is opened:** the same T with b replaced by the G response from the other seeds
  (seeds {1,2} against {3,4}, and the two other 2-against-2 splits). Its spread is the null scale; the smallest
  detectable effect is stated from it before the η result is computed.
- Rule: "separated at region level" if the mean of T over seeds is positive and more than 3.18 se from zero (t, 3
  degrees of freedom, two-sided 5%) and exceeds the largest null value. Otherwise "not separated at 4 seeds".
- J and Δ: computed the same way, description only. J's direct term is the region's own rate, so its residual
  against η is predicted to rise with the region's base rate, not with in-strength.
- Prior that the rule gives "separated": 0.6. If separated, the README line "G is not separable from the
  regional-model parameters" is narrowed to the whole-network statistics. If not, it stands with this scope.
- Also checked against the saved rates: the "effective c ≈ 1.2, r ≈ 0.5" reading of the exchange rates (17:35).

### B. The grey cells
Two different things are grey in the maps: (i) holes, where at least one of the 28 runs at step 0.01 ended non-finite;
(ii) in the cosine maps only, the quiet corner where G's effect on the seven statistics is under 3 se.
**From existing files (description).** For (ii): the size of the G column and its se across the corner. For (i): the
largest rate reached by the finite runs over the grid (`max_r`), the volume at which each of the 44 runs first went
non-finite, and the share of runs lost per cell.
**Hand estimate of the mechanism ⟦HAND, unverified⟧.** The node's Jacobian is [[2V, 2r], [J − 2π²r, 2V]]; for
r > J/(2π²) = 0.76 its eigenvalues are 2V ± i·√(2r(2π²r − J)), about 2V ± 2πr·i when r is large. An explicit two-stage
scheme is stable only while |λ|·dt stays below roughly 2, so a rate excursion to r of order 1/(2π·dt·√2) to 1/(π·dt),
about 10 to 30 at step 0.01, would make the step itself amplify. If so the failures are the scheme's, they need a large
rate excursion first, and halving the step roughly doubles the rate needed.
**Three runs that discriminate, at one hole cell (G 0.68, σ 0.054; 4 of 28 lost in the sweep), 64 new seeds
(201–264) each, on the GPU with the as-run runner:**
1. float32, step 0.01 (replication; expected about 6 to 9 of 64).
2. float64, step 0.01. Prediction: a similar share is lost (not a precision effect). Prior 0.8.
3. float32, step 0.005. Prediction: fewer than half as many are lost. Prior 0.75.
**TVB at the same cell,** 32 seeds (201–232), step 0.01, through the companion code. Prediction: at least one run is
non-finite (if TVB's rate of loss equals the JAX one, 32 runs show one with probability about 0.97). Prior 0.7. If
none, the holes may belong to the JAX code or to single precision and (2) decides between them.
**Mechanism record (laptop CPU, float32, `mprjax.make_sample` directly):** run 64 seeds at that cell and stop at the
first non-finite value, keeping the last 2000 steps of r and V. Prediction: in the region that fails first, r exceeds
10 before the failure and V oscillates with growing amplitude over the last steps. Prior 0.6 for that specific
picture. The stability estimate above stays unverified unless this record shows it.
**Horizon.** These two checks, then stop and report. Results are committed locally and not pushed until the user has
seen them.

**Amendment to check A, written at 21:25 before any per-region response was computed** (only the extraction's
row-mapping check has been seen: largest difference between a run's mean over regions and its recorded mean rate,
1.3e-7). Reason: the registered statistic T is biased under its own null. If G and η excite the same profile p across
regions, a = αp + noise and b = βp + noise; regressing a on a noisy b underestimates the multiple (regression
dilution), so the residual keeps a positive multiple of p, and p rises with in-strength. T would then come out
positive with no difference between G and η, and more so for η (smaller nudge, noisier response) than for the
G-against-G floor. ⟦HAND⟧
- **Primary statistic, replacing T for the rule:** the relative hub preference of a response,
  β(a) = [Σ_points Σ_i a_i (s_i − s̄)] / [Σ_points ā · Σ_i (s_i − s̄)²], i.e. the slope of the response on in-strength
  divided by the mean response, pooled over the grid points used. A common factor on a (signal or noise along the
  same profile) cancels in the ratio. Δβ = β(G response) − β(η response), per seed. Prediction: Δβ > 0.
- Rule, unchanged in form: "separated at region level" if the mean of Δβ over the four seeds is positive, more than
  3.18 se from zero, and larger in size than every G-against-G split difference (seeds {1,2} against {3,4}, {1,3}
  against {2,4}, {1,4} against {2,3}). Otherwise "not separated at 4 seeds". Prior unchanged (0.6).
- T is still computed and reported, with its G-against-G floor, labelled as biased.
- Also reported (description): β for J and Δ; β by side of the transition.
**Noise floor, seen at 21:27 before the η response was opened** (`src/sweep/sweep_per_region.py … floor`; 164 grid
points). Hub preference of the G response: 0.703 per unit in-strength. G-against-G differences across the three
splits: −0.0011, +0.0014, −0.0034. So a difference in hub preference of about 0.007 (1% of G's own) is the smallest
that would stand clear of the floor. The first-registered T on the same splits: +0.036, +0.035, +0.035, positive
under its null as the amendment predicted.


**Check A, outcome (2026-11-24, 21:33; `src/sweep/sweep_per_region.py` on `results/runs/sweep/node_stats_sweep.npz`,
extracted on the tower by `src/sweep/extract_node_stats.py`; 164 grid points; 4 seeds).**
- Hub preference of each response (slope of the per-region response on in-strength, divided by the mean response),
  per seed: G 0.702, 0.704, 0.706, 0.701 (mean 0.703 ± 0.001); η 0.622, 0.627, 0.632, 0.643 (0.631 ± 0.004);
  J 0.636 ± 0.004; Δ 0.626 ± 0.006.
- G minus η: +0.079, +0.076, +0.074, +0.059; mean +0.072 ± 0.005, 15 se from zero, 21 times the largest
  G-against-G split difference (0.0034). **By the registered rule: separated at region level.** G's response is
  tilted towards strongly connected regions by about 10% more than η's. Same sign and similar size on both sides of
  the transition (G ≤ 0.55: 1.004 against 0.892; G ≥ 0.57: 0.458 against 0.353).
- J and Δ sit with η (G minus J +0.067 ± 0.005; G minus Δ +0.077 ± 0.005; J and Δ within about 1 se of η). So at
  region level G separates from the three regional-model parameters, and those three do not separate from each other
  on this statistic.
- Control (post hoc, description): computing the G response over differences twice as wide changes its hub
  preference by −0.0008 ± 0.0002, so the unequal nudge sizes do not produce the effect.
- First-registered statistic T, for the record: +0.34, +0.34, +0.32, +0.34 against a G-against-G floor of +0.035. It
  is biased under its null (amendment), but ten times its floor.
- Smallest case (`src/validation/chain_response_check.py`, three regions in a chain, no noise, through
  `src/core/mprjax.py`, float64; steady rates 1.125, 1.192, 1.125): middle-to-end ratio of the response 1.544 under
  G and 0.910 under η. As predicted (prior 0.9).
- The "effective c ≈ 1.2, r ≈ 0.5" reading of the exchange rates (17:35) does NOT follow from time-averaged rates:
  at the working point the mean coupling input is 0.37, which would make one unit of η worth 2.7 of G where the map
  gives 0.80; at (0.60, 0.030) the formula gives 0.80 and at (0.56, 0.042) 1.25. The steady-state formula with mean
  rates is not the right account of the exchange rate in the switching regime. That reading is withdrawn.
- Scope and consequence. Per-region mean rates, 4 seeds, this connectome, the JAX code, σ ≥ 0.030. The whole-network
  statistics of the 17:35 entry do not separate G from η; a region-resolved statistic does, in the direction the
  equations give (prior 0.6 that it would show at 4 seeds). The README's line "G is not separable from the
  regional-model parameters" needs narrowing to the whole-network statistics. Not shown: that the paper's data could
  support such a region-resolved fit; that the 10% tilt survives the choice of connectome.

**Check B, outcome so far (21:33). TVB and the float64 job are still running.**
- Kind (ii), the quiet corner: not yet tabulated.
- Kind (i), holes. In the sweep, losses by σ row at step 0.01: none of 868 runs up to σ 0.038; 5, 3, 9 and 24 of
  868 at σ 0.042, 0.046, 0.050, 0.054; all at G ≥ 0.55. First non-finite volume spread over the whole record
  (quartiles 125, 276, 542 of 833; 4 of 44 in the first 50): not a burn-in effect. Lost runs are spread evenly over
  the four seeds (10, 12, 11, 11). **The 44 lost runs are 35 distinct events:** runs at different settings that share
  a seed, and so share the noise, fail at the same volume (seed 4, volume 225: four settings from G 0.64 to 0.70 and
  σ 0.046 to 0.054). The trigger is a particular stretch of noise.
- The largest output-sample rate reached by finite runs grows towards the same corner: about 2 to 4 at low σ, 6 to
  15 at σ 0.054 and high G.
- **Replication at the hole cell (G 0.68, σ 0.054; 64 new seeds each; `results/runs/validation/hole_cell/`):**
  float32 at step 0.01: 6 of 64 lost (prediction 6 to 9). float32 at step 0.005: 3 of 64 lost. The prediction "fewer
  than half as many" is not met (3 is half of 6), and 3 against 6 is not a resolvable difference at this size. The
  three seeds lost at the half step (209, 261, 263) are among the six lost at the full step, at different volumes;
  the chance of that overlap is about 1 in 2000, and it is not explained (the noise draws differ between the two
  steps; the starting voltages are shared).
- **Mechanism record (`src/validation/blowup_record.py`, laptop CPU, float32, seeds 201–264; first failure: seed 207
  at 3846.6 time units; `results/runs/validation/blowup_record_f32.npz`).** The region that fails is the most
  strongly connected one (in-strength 4.90). In the 400 steps before, it swings widely (r between 0.3 and 3, V
  between −2.3 and +2.5) while the network sits half up. About 80 steps before the end its rate is at the zero bound
  while its V is positive (+1.7) and rising. The rate then stays between 0.04 and 0.2 for 60 steps while V climbs
  without turning: 2.9, 4.0, 6.5, 10.6, 18, 39, 60, 114, 185. Only in the last 8 steps does the rate follow (0.5,
  1.2, 2.3, 5.4, 20, 178); the next step overshoots to V = −2761 with r at zero, and two steps later the values are
  infinite. The state "V above 2 with r below 0.3" does not occur once in the preceding 2460 steps of any region.
- **The registered picture was only partly right.** The rate did exceed 10 before the failure, but only in the last
  two steps, and V did not oscillate with growing amplitude: it ran away upward for about 60 steps and overshot once.
  The eigenvalue estimate registered above is not what the record shows and stays unverified.
- Reading ⟦HAND, unverified⟧. With the rate near zero, the rate equation's growth term 2rV has nothing to multiply,
  and the per-step noise on r (sd 0.033) is ten times its drift (Δ/π·dt = 0.003), so r stays small; meanwhile
  V̇ ≈ V² + η + G·c grows without the −π²r² term that normally turns it. The equations then produce a spike whose
  height is set by how small the rate was when V turned positive; once V·dt approaches 1 the explicit step cannot
  follow it. On this reading the losses come from additive noise on a rate held at a zero bound, in a strongly driven
  region; a smaller step raises the ceiling but does not remove the spikes. Other runs in the same batch reached
  rates of 28.7 and survived.

**Check B, further outcome (2026-11-24, 21:50; TVB at 30 of 32 runs, the last two still running).**
- **Quiet corner (kind ii).** 51 of 310 grid points have a combined G effect under 3 se, and 5 more have no estimate
  beside a hole. They are G ≤ 0.48 to 0.53 for σ ≤ 0.030, most of the σ 0.018 row, and a few neighbours of holes.
  There the network sits in the low state: up-occupancy under 1.5%, mean rate 0.10 to 0.13, and one G notch moves the
  mean rate by about 0.0002 (at (0.45, 0.022): +0.00017 ± 0.00006). The median up-fraction is exactly zero in all four
  runs at 25 points. Nothing fails there; there is nothing to compare.
- **float64 at the hole cell:** 9 of 64 lost (float32: 6 of 64). Prediction held (prior 0.8): not a precision effect.
  The lost seeds differ from the float32 ones, as expected from different noise.
- **TVB at the hole cell:** 2 of 30 lost so far (seeds 226, 227). Prediction held (prior 0.7): TVB loses runs at this
  setting too, so the holes belong to the scheme and settings, not to the JAX code. Rates of loss: TVB 2 of 30, JAX
  float32 6 of 64, JAX float64 9 of 64; not distinguishable at these sizes. Mean rate of the finite runs: TVB 0.5833
  ± 0.0003, JAX float32 0.5835 ± 0.0002, a further point at which the two codes agree.
- **Summary of what the holes are.** Rare events (a few percent of runs per 60,000 time units at σ 0.054, none seen
  below σ 0.042) in which a strongly connected region's potential runs away while its rate is held near zero, ending
  in a spike the explicit step cannot follow. They occur in TVB and in JAX, in single and double precision, and are
  not removed by halving the step (3 of 64 against 6 of 64, unresolved). They are set off by particular stretches of
  noise. Not established: that the zero bound is necessary for them (one recorded case); how the rate of loss scales
  with the step; why the half-step losses fall on seeds that also fail at the full step.
- **Horizon reached** for both checks. Readability rewrite of the core code done in the same session (commit
  `c8ae1f3`), verified against the same checks.

**Check B, TVB complete (21:52).** All 32 TVB runs at (G 0.68, σ 0.054) are in: 2 of 32 non-finite (seeds 226,
227). The other counts are unchanged: JAX float32 6 of 64, JAX float64 9 of 64, JAX float32 at half the step 3 of 64.
Mean rate of the 30 finite TVB runs 0.5833 ± 0.0003 (JAX float32 0.5835 ± 0.0002).

## 2026-11-25 (07:23), empirical test of the confounding, registered before running
User: "we should be able to derive the confounding maps from the derivative maps and they should line up with the
empirically observed confounding maps no?" As built they line up by construction: each cosine in the confounding maps
is computed from the derivative maps at that grid point. An independent check needs runs, not derivatives.
**Test (existing runs only).** If a nudge in η (or J, Δ, or halving the step) is equivalent to a shift in G, the runs
made at the nudged setting should sit ON the base curve along G, in every statistic at once.
- For each grid point (mixed region, σ ≥ 0.030, no holes) and each nudged setting: G* = the G at which the base curve
  of the mean rate (piecewise linear between grid values, same σ) equals the nudged mean rate. Then for VAF, mean FC,
  corr(FC, SC) and the switching index: z = (nudged value − base curve at G*) / se, with the se combining the nudged
  runs, the interpolated base runs, and the uncertainty of G* itself.
- Calibration (the null): the same procedure applied to the float64 base row at σ 0.030 against the float32 base
  curve. Those are independent realisations of the same setting, so their z values show what "on the curve" looks
  like under this procedure with 4 seeds.
- **Prediction 1:** the empirical exchange rates, (G*₊ − G*₋)/(2δ), have medians within 15% of the derivative-based
  ones (η 0.85, J 0.41, Δ 1.02 of G per unit). Prior 0.85. (Partly forced: both use the mean rate.)
- **Prediction 2:** for VAF and mean FC the share of |z| < 2 among the nudged settings is within 10 percentage points
  of the calibration's share. Prior 0.7. If it is clearly lower, the nudges are NOT pure G shifts for those
  statistics, and the whole-network confounding is only first-order.
- Larger nudges (pilot, 16 seeds, working point; η ± 0.01, 0.02; J ± 0.05, 0.10; Δ ± 0.01, 0.02), same procedure
  against the σ 0.030 base curve: description, with a prior of 0.5 that all four statistics stay within 2 se.
- The step: the same for the base runs at step 0.005. Description.

**Outcome (07:25; `src/sweep/sweep_matching.py`; 189 grid points; existing runs only).**
- Calibration (float64 base row against the float32 base curve, 17 points; true shift zero): matched G off by +0.0008
  (median), sd 0.0009. Share of |z| < 2: VAF 1.00, mean FC 0.94, corr(FC, SC) 0.71, switching index 0.88. Only 17
  points, so these shares are loose.
- **Prediction 1 held.** Empirical exchange rates (medians): η 0.877, J 0.413, Δ 1.023 of G per unit, against the
  derivative-based 0.85, 0.41, 1.02 (ratios 1.03, 1.01, 1.00). Partly forced, as noted.
- **Prediction 2 held for mean FC and failed for VAF.** Share of |z| < 2: mean FC 0.96, 0.91, 0.97 for η, J, Δ
  (calibration 0.94); VAF 0.88, 0.72, 0.83 (calibration 1.00), i.e. 12, 28 and 17 points lower. By the registered
  reading, the nudges are not pure G shifts for VAF.
- **POST HOC, description: the VAF mismatch has a consistent sign.** Below the transition (G ≤ 0.55), a setting
  reached by raising η, J or Δ has LOWER VAF than the base curve at the G with the same mean rate, and one reached by
  lowering them has HIGHER VAF: mean z −1.34 ± 0.09 and +1.57 ± 0.10 for η; −2.39 ± 0.15 and +2.41 ± 0.14 for J;
  −1.50 ± 0.11 and +1.71 ± 0.10 for Δ (about 85 points each; points share base cells, so the se understate). In
  VAF's own units: −0.0027, −0.0050, −0.0031 for the upward nudges, where the base VAF changes by about 0.004 to
  0.010 per G notch. Mean FC and corr(FC, SC) show the same signs at about a third of the size; above the transition
  everything is smaller (VAF ∓0.2 to 0.5). The opposite signs for opposite nudges rule out interpolation error and
  shared base-cell noise, which would act in one direction.
- Reading ⟦HAND, unverified⟧: at equal mean rate, a network with more coupling and less drive is more coordinated.
  G buys coordination with activity; η, J and Δ buy activity with less of it. This is the whole-network trace of the
  region-level tilt found on 2026-11-24.
- **Consequence for the 17:35 statement** ("no separation of η, J or Δ from G is shown anywhere, and none is excluded
  below the noise"): a small one is now shown, by a directed test pooled over points. The exchange rates stand; the
  directions are close to parallel (cosines about 0.99) but not identical. The split-half ceiling could not see it
  because it is small against single-point noise.
- Half the step against the base curve: matched G +0.0051 (median). VAF sits below the curve (mean z −2.04, 59%
  within 2), mean FC too (−1.09). So halving the step is not a pure G shift either; it acts more like raising η.
- Larger nudges at the working point (pilot, 16 seeds, against a base curve from 4): downward nudges that move the
  matched G to 0.47–0.52 give z of +2.5 to +3.9 on VAF, mean FC and the switching index; upward nudges stay within
  2. Same sign pattern, but all rows share one noisy base curve (the base setting itself reads +2.9 on the switching
  index), so this is not independent evidence.

**Presentation changes (2026-11-25, 07:38; user requests).**
- The grey-cell analysis has its own note, `doc/grey_cells.pdf` (`src/sweep/writeup_grey_cells.py`); the sweep
  write-up points to it. The two builders share `src/sweep/report_common.py`.
- Section 5 of the write-up now shows the derivative maps, says how they are built, and states that the confounding
  maps are computed from them (steps, and one cosine worked out at the working point).
- **Deviation from the registered display rule, display only.** The 12:05 registration said a derivative is shown
  only where it exceeds 2 se and is blank elsewhere. From today every estimate is drawn, and the ones under 2 se (for
  the cosines: where G's own effect is under 3 se) are faded, not blanked; grey is kept for cells with no estimate.
  Reason (user): the unresolved cells still carry shape. The summaries and counts use the registered rules unchanged.
- Description, post hoc: in the 66 cells whose cosine is drawn faded, the cosine of G with η, J and Δ has median
  +0.92, +0.95, +0.92 and is positive in 98%, 100%, 100% of cells.

## 2026-11-25 (07:41), a hand account of the per-region response, and a check of it, registered before running
User: "the regions share in response looks logistic for eta and G? Can we hand derive what the response should be
and see what falls out?"
**Hand derivation ⟦HAND, unverified⟧.** Treat each region as a two-state unit (down, up) that noise flips, with flip
rates k↑(I), k↓(I) set by its input I. Three assumptions:
(a) log(k↑/k↓) is linear in the input over the range that matters (barriers move linearly with input; rates are
    exponential in barrier over noise). Then the share of time up is logistic: p = 1 / (1 + exp(−κ (I − I*))).
(b) A region's input is its drive plus the coupling computed from the OTHER regions' time-averaged rates (mean-field
    closure; ignores that regions switch together): I_i = η + G c_i, c_i = Σ_j W_ij m_j.
(c) Its mean rate is m_i = r_d + (r_u − r_d) p_i, with fixed down and up rates.
Consequences:
1. **Across regions at one setting, logit(p_i) is linear in c_i**, with slope κG. Since c_i grows with in-strength,
   occupancy against in-strength is a logistic curve.
2. **Response to a parameter.** Differentiating, δm = χ ∘ (b δθ + G W δm), with χ_i = (r_u − r_d) κ p_i (1 − p_i), so
   δm/δθ = (1 − G X W)⁻¹ X b, X = diag(χ). The parameter enters only through b: b = 1 for η, b = c for G.
   χ is bell-shaped in the input: regions that are always down or always up do not respond.
3. **The pooled G response is a logistic that saturates.** Summing ∂m_i/∂G over a range of G telescopes to
   m_i(G_max) − m_i(G_min) = (r_u − r_d) [p_i(G_max) − p_i(G_min)]. If everything starts down, that is the logistic
   p_i at the top of the range: strongly connected regions go from fully down to fully up, the most any region can
   change, so they sit on a plateau. (Exact arithmetic given (c).)
4. **Why η looks like G.** When the largest eigenvalue λ of G X W is near 1 (the network switches collectively), the
   inverse is dominated by one mode: δm ≈ u (v·X b) / (1 − λ). The profile u over regions is the same whatever b is;
   b only sets the amplitude. That is the confounding, and the exchange rate is (v·X 1)/(v·X c) = 1/c̄, with c̄ the
   coupling input averaged over the regions that are switching (weights v_i χ_i), not over all regions.
5. **Why G is tilted to hubs.** The other modes keep the imprint of b. b = c rises with in-strength and b = 1 does
   not, so G's response is more hub-weighted, by an amount that shrinks as λ → 1.
The repository's earlier two-state study (`src/earlier_studies/two_state_reduction`) fitted flip rates against input
in the same spirit; that work used the superseded step 0.05 settings. Prior art for linear response of noisy bistable
networks was NOT searched; this is a hand account, not a claim of novelty.
**Check on the saved per-region numbers (`src/sweep/sweep_two_state.py`; base runs, 4 seeds averaged; grid points of
the per-region analysis).**
- T1: at each grid point, regress logit(up-occupancy) on c_i over regions with occupancy between 0.002 and 0.998.
  Prediction: median R² above 0.9. Prior 0.6.
- T2: at each grid point take κ from that slope and (r_u − r_d) from the regression of m_i on p_i; compute the
  predicted responses (1 − G X W)⁻¹ X b for η and G; pool them exactly as the measured ones. Predictions: (i) predicted
  hub preference is larger for G than for η (prior 0.8); (ii) both within 20% of the measured 0.703 and 0.631 (prior
  0.4); (iii) predicted exchange rate, the ratio of pooled mean responses, within 30% of the measured 0.85 (prior 0.4).
- T3 (description): the plateau height in the pooled G profile against (r_u − r_d) / mean change; the largest
  eigenvalue of G X W over the grid.
If T1 fails the two-state picture is wrong at the first step and the rest is not read.

**Outcome (07:42; `src/sweep/sweep_two_state.py`; 164 grid points).**
- **T1 held.** logit(share of time up) against coupling input across the 84 regions: R² quartiles 0.90 / 0.94 / 0.98,
  above 0.9 at 73% of grid points, rising with σ (median 0.87 at σ 0.030, 0.98 at σ 0.054). Occupancy is close to a
  logistic function of a region's input. Up rate minus down rate 1.13, down rate 0.13.
  Not as assumption (a) would have it: the fitted κ is about 4 to 4.7 at every σ, where barrier-over-noise rates
  would make it fall as σ rises. The logistic is a good description; the Arrhenius reading of its slope is not
  supported.
- **Consequence 3 holds as arithmetic and in the data.** In the pooled G profile the regions with in-strength above
  2.7 sit at 2.12 times the mean response and the ten weakest at 0.03: the saturating curve.
- **T2 failed on all three counts.** Predicted hub preference 0.947 for G and 1.018 for η (measured 0.703, 0.631):
  the predicted difference has the wrong sign (−0.072 against +0.072). Predicted exchange rate 0.53 against 0.875.
  The pooled predicted responses even have the wrong overall sign.
- **Why (T3, description).** The fitted feedback gain, the largest eigenvalue of G X W, exceeds 1 at 43% of grid
  points (quartiles 0.74 / 0.93 / 1.17). Along σ 0.030 it is 1.8 to 2.2 at G 0.53–0.55 (2.23 at the working point),
  falls through 1 near G 0.59, and is 0.44 at G 0.69. Where it exceeds 1 the formula (1 − G X W)⁻¹ has no meaning:
  the mean-field closure (b) says the averaged state is unstable there.
- POST HOC, description: those points are the transition band (G 0.48–0.59), where the per-volume share of regions
  up fluctuates most relative to its mean (sd/mean 0.43 against 0.22–0.26 elsewhere). Split by gain: below 0.8 (52
  points) the formula gives the size of the responses to within about 20% (1.22 and 1.08 of measured), profiles that
  correlate with the measured ones at 0.95 and 0.92 per point, an exchange rate of 0.79 against 0.89, and the right
  sign for the hub tilt (+0.026 against +0.103), with both hub preferences too high (0.60 and 0.57 against 0.38 and
  0.27). Above 1 (70 points) predicted and measured profiles are anticorrelated.
- **What falls out, in plain terms.** The logistic shape is right and has a simple origin: a region's occupancy is a
  logistic function of its input, and the pooled response to G is the total change of each region across the range,
  which saturates for strongly connected regions. A linear-response formula built on that works away from the
  transition and fails inside it, because there the regions switch together and a region's average input is not
  what it responds to. The band where it fails contains the paper's working point.
- Reading ⟦HAND, unverified⟧ for a next account: in the band the network has two collective states and the averages
  are a mixture of them. A parameter then acts mostly by changing the share of time spent in each, which would give
  the same regional profile for every parameter (the difference between the two states) and so the confounding, by a
  different route from consequence 4. Not tested.

**Essay v5 (2026-11-25, 08:03; user: "go ahead and proceed with those recommended edits").** `doc/essay/v5/essay_v5.tex`
is v3 reordered; v3 is untouched. Assembled mechanically from v3's lines, then checked: every non-blank v3 line is in
v5 unchanged apart from cross-references and marks (the only exceptions are the author and date lines), and every
other line sits inside the coloured new-text marker (about 1,450 words: a status note, "The argument in brief", §3.3
a small case from the v4 theory draft, §3.4 a worked case from the sweep, one sentence before Table 1, two appendix
notes).
- Moved: v3 §3.4–3.6 (the nine statistic-by-class cases and Figure f2) to Appendix B; the gate table is Appendix C.
  Kept in the body: the Proposition with its proof, the three classes, Table 1 and the paragraph that reads it.
- The 17 passages the reversal check lists (E: 210, 237, 255, 323, 351, 413; R: 216, 243, 249, 260, 273, 279, 285,
  291, 305, 348, 414; description superseded: 255, 279, 323, 413; v3 line numbers) carry `\REV{}` marks. They are
  marked, NOT corrected.
- Not done: open item 0 (the unbounded-condition-number wording against the audited note; Rothenberg and
  Transtrum–Qiu). Not done: section summaries. Offered, not done: moving the Proposition's proof to an appendix.
- The new §3.4 states measured values only (η ± 0.02 with matched G 0.521 and 0.556; exchange rates; region-level
  tilt; the post hoc VAF difference) and claims nothing as new. "Audit before drafting" was set aside on the user's
  explicit go-ahead; the section is illustration, unaudited, machine-drafted, and marked as such.

### 2026-11-25, 09:38, write-up wording and labels (user's questions)

- "Chain" is no longer used for a simulation. A run is one simulation (one setting, one seed); runs that share a
  setting are replicates. "Chain" is reserved for MCMC, of which there is none here. Section 2 is now "Runs are still
  settling after the discard"; the figure is `w2_runs_over_time.png`.
- "What was run" has a terms note and a table of how many runs stand behind each kind of value (statistic, each
  derivative, step effect, double-precision row, TVB comparison).
- Derivative maps: the row title used to sit on the axis name ("η + 0.005" over "σ") and read as "η ± σ". Rows now
  read "change when η goes up 0.005"; the axis reads "noise level σ"; the caption says σ is a model parameter, not a
  standard deviation. No number changed.

### 2026-11-25, 09:53, the lab's container (`lrokos/tvb_node:latest`): what its public metadata shows

The user pointed to the workshop instructions (`INN-SFU/tvb-node`, `Pre-Session_Materials`), which use this image.
The image was **not pulled and not run**. Read: its manifest and 7 kB configuration record from Docker Hub, and
public files on GitHub and PyPI. Nothing inside the image's layers was read.

Observed.
- Image index `sha256:97bc26ea…3156`; amd64 image `sha256:8c8e8a32…18c7`, built 2026-01-23, 10 layers, 1.25 GB
  compressed. Build steps: a miniconda base; `conda create -n tvb_node_env python=3.7`; `COPY virtual_aging_brain`;
  `pip install -r virtual_aging_brain/requirements.txt`; `pip install` that folder; jupyterlab.
  `PYTHONPATH=/tvb_node/virtual_aging_brain`.
- A public repository of that name exists: `ins-amu/virtual_aging_brain` (MIT, last pushed 2023-03-10). It holds
  the helper package the notebooks import (`src/simulation.py`, `src/analysis.py`, `src/data.py`), which the earlier
  entry recorded as "not in the repository". Its `requirements.txt` pins `tvb-library<=2.5` and lists `sklearn`;
  the image sets `SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL`, which fits that line.
- `src/simulation.py` there has `run_nbMPR_backend` (TVB's `NbMPRBackend`), `configure_conn(conn_file, conn_speed)`,
  and stochastic Heun with noise `[σ/τ, 2σ]`.
- PyPI has `tvb-library` 2.5 (2022-05-15, no Python restriction); 2.10.0 needs Python ≥ 3.8.
- TVB's MPR kernel, compared between release tags (licence header ignored): the integration template
  `nb-montbrio.py.mako` and `nb-dfuns.py.mako` have **no differing code line between 2.5 and 2.10.0**. The wrapper
  `nb_mpr.py` differs in 37 lines, which by my reading only move the call into a helper and add a `print_source`
  option. At 2.4 the template is a different, older implementation.

Inferred, not validated.
- The container most likely runs `tvb-library` 2.5: that is what pip resolves from `<=2.5` if the image's copy of
  the requirements file is the public one. The image's copy was not read.
- If so, its MPR integration code is the same as in 2.10.0, the version our TVB reference used.

What would settle it: `pip show tvb-library` inside the container (the user's to run), or a diff of the image's
`virtual_aging_brain` folder against the public repository.

### 2026-11-25, 10:14, write-up: the derivative maps in symbols (user's request)

Section 5 now states in notation what a derivative map is: the run mean and its standard error, the central
differences for η, J, Δ (nudges) and for G, σ (neighbouring grid points), the step difference, the definition of a
notch, and the panel value M = D · δ. General statement only, no worked numbers. `report_common.py` gained a
`("latex", source)` block for formulas. No number changed.

### 2026-11-25, 10:21, write-ups: informal terms replaced by standard ones (user's request)

| Was | Now | Conflict avoided |
|---|---|---|
| notch | increment (δ_p); grid spacing for G and σ | none |
| nudge, nudged | perturbation, perturbed | none |
| the step, half step, grid step | time step (integration); grid increment | "step" meant both |
| discard, settling | initial transient removed; not yet stationary | "burn-in" reads as MCMC |
| noise (of four runs), noise-widths | sampling variability; run-to-run standard deviations | "noise" is also the model parameter σ |
| noise level σ | noise amplitude σ | σ read as a standard deviation |
| knee | steep part of the transition | none |
| confounding (maps) | collinearity (maps): cosine similarity of sensitivity vectors | "confounding" is a causal-inference term |
| exchange rate | G-equivalent (the least-squares coefficient) | "rate" is also the firing rate |
| list, pushes, undone | vector, moves, compensated | none |
| quiet cells; lost runs | unresponsive cells; diverged runs (non-finite values) | none |
| high / low state | up / down state | none |

Applied to `doc/sweep_writeup.pdf`, `doc/grey_cells.pdf`, their figures, `README.md` and `src/README.md`. The terms are
defined once in "What was run". File names, variable names in the run files, the essay drafts and earlier log
entries keep the old words. No number changed.

### 2026-11-25, 10:36, collinearity in symbols, with a machine check of the algebra (user's request)

Section 5 of the write-up now states, before the collinearity figure, what the collinearity map is in notation: the
scaled sensitivity vectors a and b, the condition "some non-zero change leaves every statistic unchanged", its
equivalence to b = λa and to |cos(a, b)| = 1, the G-equivalent λ = a·b/‖a‖², the residual r = b − λa with
‖r‖² = ‖b‖²(1 − cos²), and why one statistic can never separate two parameters. General statement only.

**Machine check.** `doc/collinearity_check.lean` (Lean 4.29, Mathlib v4.29.0; compiled with no error against the
shared Lean project of `~/Research/Mathematics/Resolvent_Framework`; axioms propext, Classical.choice, Quot.sound for
all six). These are textbook facts, checked because the write-up leans on them.

Statement card (the write-up's sentence beside the Lean statement; **the read of this card is the user's and is owed**):

| Sentence in the write-up | Lean statement |
|---|---|
| With one statistic the condition always holds | `one_statistic (a b : ℝ) : ∃ x y, (x, y) ≠ (0, 0) ∧ a * x + b * y = 0` |
| With two, the condition is a₁b₂ − a₂b₁ = 0 | `two_statistics : (∃ x y, (x, y) ≠ (0, 0) ∧ a₁ * x + b₁ * y = 0 ∧ a₂ * x + b₂ * y = 0) ↔ a₁ * b₂ - a₂ * b₁ = 0` |
| a ΔG + b Δp = 0 for some non-zero change ⇔ b = λa | `unseen_iff_multiple (a b : E) : a ≠ 0 → ((∃ x y, (x, y) ≠ (0, 0) ∧ x • a + y • b = 0) ↔ ∃ l, b = l • a)` |
| b = λa ⇔ \|cos(a, b)\| = 1 | `abs_cos_eq_one_iff (a b : E) : a ≠ 0 → b ≠ 0 → (\|⟪a, b⟫ / (‖a‖ * ‖b‖)\| = 1 ↔ ∃ l, b = l • a)` |
| ‖r‖² = ‖b‖²(1 − cos²(a, b)), r = b − λa, λ = a·b/‖a‖² | `residual_sq_cos (a b : E) : a ≠ 0 → b ≠ 0 → ‖b - (⟪a, b⟫ / ‖a‖ ^ 2) • a‖ ^ 2 = ‖b‖ ^ 2 * (1 - (⟪a, b⟫ / (‖a‖ * ‖b‖)) ^ 2)` |

- Symbols: `E` is any real inner-product space (ℝ⁷ in the write-up); `x`, `y` are ΔG, Δp; `⟪a, b⟫` is a·b.
- Totalised operations: division by ‖a‖‖b‖ and by ‖a‖². In Lean a division by zero gives 0; the hypotheses a ≠ 0,
  b ≠ 0 exclude it, and the write-up states "for a ≠ 0 and b ≠ 0". In the data, the rule that G's vector exceed three
  standard errors keeps a away from 0.
- Not covered by the Lean file: that λ is the least-squares coefficient (it minimises ‖b − la‖; standard, by hand);
  that a first-order statement says anything beyond first order; the choice of scaling by sd.

**The same on the data** (not a proof; the instantiated case). At the working point, for η over the seven statistics:
cos = +0.9574, G-equivalent 0.796, and ‖r‖² equals ‖b‖²(1 − cos²) to 2 × 10⁻¹⁰. Taking the statistics one at a time,
|cos| = 1 for each of the seven there, and for all 2,033 statistic-cells of the grid with both slopes non-zero.

`report_common.py` now loads `amssymb`. No number in the report changed.

### 2026-11-25, 10:40, collinearity "In symbols" passage taken out of the write-up (user's decision)

The user is not yet confident of the mathematics or of what it implies for the analysis, and will read the statement
card first. The passage added in f4d00ce (the two paragraphs before the collinearity figure) is removed from
`doc/sweep_writeup.pdf`. The plain-words derivation, the working-point table and the figure stay. The derivative-map
"In symbols" passage stays. `doc/collinearity_check.lean` and the card above are kept as the material to be read. The
removed text is in the history at f4d00ce. Nothing is to be built on the collinearity algebra until the card is read.

### 2026-11-25, 13:42, the essay is parked; theory in the note, numbers here (user's decision)

- The user proposed deleting the essay and working between the numbers in this repository and the theory in
  `~/Research/Statistics/identification-through-reduction/`. It was archived instead of deleted, on the user's word:
  `doc/essay/` is now `archive/essay/` (`archive/README.md` says what it is and what only it holds). The 22 tracked
  files moved with their history; the drafts' backups, which exist only on disk, moved with them.
- The paper's extracted text moved from `doc/essay/v3/audit/` to `external/fousek2024/`. It is third-party and stays
  on disk only (ignored by git).
- Owed item 0 ("bring the essay in line with the note") no longer applies and is marked so above.
- Still without a home: what the numbers say about the paper's claim. The note is domain-free and the sweep write-up
  reports numbers. Left for a later page in the user's words.
- Path translation for older entries: `doc/essay/…` and `essay/…` → `archive/essay/…`.
