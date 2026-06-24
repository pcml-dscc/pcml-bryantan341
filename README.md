[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-2e0aaae1b6195c2367325f4f02e2d04e9abb55f0b24a779b69b11b9e10269abc.svg)](https://classroom.github.com/online_ide?assignment_repo_id=24173319&assignment_repo_type=AssignmentRepo)
# MLFP — Student Repository

This is your **personal copy** of the MLFP course materials, created for you by
GitHub Classroom. Everything below applies to your copy; the course template at
`https://github.com/pcml-dscc/mlfp` is the "upstream" you pull updates from.

---

## What's in this repo

| Path                                       | What it is                                                                       |
| ------------------------------------------ | -------------------------------------------------------------------------------- |
| `modules/mlfpNN/readings/deck.pdf`         | Full slide deck (read this first for each module)                                |
| `modules/mlfpNN/readings/textbook.pdf`     | Long-form textbook (deep reading)                                                |
| `modules/mlfpNN/readings/notes.pdf`        | Condensed notes (revision)                                                       |
| `modules/mlfpNN/local/ex_N/`               | Your exercise files (edit these in VS Code)                                      |
| `modules/mlfpNN/colab-selfcontained/ex_N/` | Self-contained Colab notebooks — open directly in Colab, no git clone needed     |
| `modules/mlfpNN/solutions/ex_N/`           | **Reference solutions** — consult AFTER attempting your own                      |
| `modules/mlfpNN/diagnostic-reference/`     | Captured diagnostic outputs (plots + reports) for reference                      |
| `shared/`                                  | Helper code the exercises import (diagnostics, data loaders, etc.) — do not edit |
| `data/`                                    | Small datasets (large ones auto-download on first run)                           |

---

## One-time setup (VSCode terminal)

Open the VSCode integrated terminal (`` Ctrl+` `` or `` Cmd+` ``) and paste:

```bash
# 1. Confirm you're inside YOUR fork
git remote -v

# 2. Add the instructor's repo as "upstream" — copy this ONE line
git remote add upstream https://github.com/pcml-dscc/mlfp.git

# 3. Verify both remotes
git remote -v
# origin    → your personal copy
# upstream  → pcml-dscc/mlfp (course template)
```

That's it — setup done once per machine.

---

## Pull the latest from your instructor (do every class)

In the VSCode terminal:

```bash
# Save your own work first
git add -A && git commit -m "wip" || true

# Fetch + merge from instructor
git fetch upstream && git merge upstream/main

# Push the merged result back to your fork
git push origin main
```

If you see merge conflicts: VSCode highlights them with **Accept Incoming /
Current / Both** buttons in each conflicted file. Choose:

- **Accept Incoming** for any files in `shared/**`, `pyproject.toml`,
  `uv.lock`, `data/**` (these are instructor-managed)
- **Accept Current** or resolve by hand for files YOU edited (usually your
  exercise solutions under `modules/mlfpNN/local/`)

---

## Running exercises locally

```bash
# One-time — install dependencies
uv sync

# Run an exercise — Modules 1-4 are single files:
.venv/bin/python modules/mlfp01/local/ex_1.py

# Modules 5-6 split each exercise into a directory of technique files:
.venv/bin/python modules/mlfp05/local/ex_1/01_standard_ae.py
```

---

## Running exercises in Google Colab

Each exercise has a self-contained `.ipynb` notebook under
`modules/mlfpNN/colab-selfcontained/ex_N/`. No git clone, no `FORK_URL`,
no `sys.path` setup — all helpers are inlined.

1. Download the `.ipynb` from your fork (or drag-and-drop from your file manager).
2. Open it in [Google Colab](https://colab.research.google.com) — File → Upload notebook.
3. **Before running**: Runtime → Change runtime type → **T4 GPU** (free tier, needed for Modules 5-6).
4. Run **Cell 0** — installs `kailash`, `kailash-ml`, `kailash-kaizen`, `torch`, etc.
5. Run **Cell 1** — defines all shared helpers. You can collapse this cell with the ▼ arrow.
6. Fill in the `____` / `# TODO` blanks in subsequent cells and run them.

### Pulling updates

When your instructor pushes updates, re-download the specific notebook you're
working on. Your in-Colab edits are NOT overwritten — they live in the Colab
cloud copy until you save a Drive version. Download a fresh template whenever
you want to restart an exercise from scratch.

---

## Where to find the diagnostic outputs

Module 5 exercises use the `DLDiagnostics` toolkit to audit your trained
models. You'll see its output in three places:

1. **Inline in each exercise file** — search for
   `# ══════ EXPECTED OUTPUT ══════` to see what a healthy (or unhealthy)
   diagnostic report looks like
2. **In your terminal/notebook output** — when you run the exercise, the
   `diag.report()` call prints the live Prescription Pad
3. **In `modules/mlfp05/diagnostic-reference/`** — captured reports and
   interactive Plotly dashboards from reference runs. Open the `.html` files
   in any browser to see a healthy training run's 4-panel dashboard.

To see the live dashboard for **your own** run, change `show=False` to
`show=True` in the diagnostic call, or use:

```python
diag.plot_training_dashboard().show()
```

---

## Help / Troubleshooting

| Problem                              | Fix                                                                                                |
| ------------------------------------ | -------------------------------------------------------------------------------------------------- |
| `ModuleNotFoundError: shared.mlfp05` | Locally: run `uv sync`. In Colab: re-run Cell 1 (inlined helpers) from the top                     |
| Merge conflicts on `git pull`        | Keep your answers; take instructor's version for files in `shared/**`, `pyproject.toml`, `uv.lock` |
| "Authentication failed" on push      | Use a GitHub Personal Access Token, or set up `gh auth login`                                      |
| Colab says "No module named torch"   | Cell 0 hasn't finished — wait for pip install to complete, then re-run Cell 1                      |
| PDF looks stale after `git pull`     | PDFs are binary — git pulls them correctly. If you opened one before pulling, close & reopen       |

---

## Support

- Questions about exercises: post in the class Slack / LMS
- Bug in course materials: open an issue on the instructor repo
- Can't resolve a merge conflict: ask your TA or paste the conflict into Slack
