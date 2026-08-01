# logbook_info.md

Instructions for Claude Code running in the Brev session. Read this file, then
produce an engineering status update for the thesis logbook.

Copy this file into the repo root on Brev. To trigger it, say something like:

> Read logbook_info.md and generate this week's update.

---

## What this is for

The thesis logbook lives off-machine (`main.tex`, maintained in a separate
session). That session has no visibility into what actually happened on the
Brev box — commits, runs, crashes, results. This file is the handoff.

Your job is to produce a **dense, factual record of what happened on this
machine** since the last update. A separate process compresses it into the
logbook's word-limited sections and updates the Linear board. So:

**Do not compress. Do not write prose for a marker. Do not editorialise.**
Detail that looks excessive here is what makes the logbook accurate later.

---

## Before writing: gather evidence

Actually run these. Do not reconstruct from memory or from the conversation.

```bash
# When was the last update generated?
ls -la update_*_eng.md 2>/dev/null | tail -5

# Commits since then (adjust --since to the last update's date)
git log --since="YYYY-MM-DD" --pretty=format:'%h %ad %s' --date=short --stat

# Uncommitted work
git status --short && git diff --stat

# Result artefacts produced this period
find . -newermt "YYYY-MM-DD" \
     \( -name '*.png' -o -name '*.csv' -o -name '*.json' -o -name '*.npy' -o -name '*.pt' \) \
     -not -path './.git/*' | head -50

# Experiment logs / run outputs
ls -lat logs/ runs/ outputs/ results/ 2>/dev/null | head -40

# Environment state — this project has a history of environments vanishing
nvidia-smi --query-gpu=name,memory.total --format=csv
pip list 2>/dev/null | grep -iE 'torch|transformers|robosuite|libero|openvla'
df -h . | tail -1
```

If a command returns nothing, that itself is a finding — record it.

---

## Output

Write one file in the repo root:

```
update_YYYY-MM-DD_eng.md
```

`YYYY-MM-DD` is today's date (`date +%F`), sortable so updates list in order.
The `_eng` suffix marks this as the engineering-side update, leaving room for
other sources later.

Print the full path when done so it can be pulled off the box.

---

## Template

Fill every section. If a section genuinely has nothing, write
`Nothing this period.` — **never invent activity or results to fill space.**

```markdown
# Engineering Update — YYYY-MM-DD

**Period covered:** YYYY-MM-DD to YYYY-MM-DD
**Repo / branch:** <name> @ <branch>
**HEAD:** <short SHA>
**Instance:** <GPU, disk free, whether the instance was rebuilt this period>

## 1. Headline

Two or three sentences. What actually moved. If the honest answer is "the week
went on infrastructure and no research progress was made", say exactly that —
that is a legitimate and useful logbook entry.

## 2. Work completed

Bullets. Each one carries its evidence: commit SHA, file path, or the command
that produced it.

- <what was done> — `<sha>` / `path/to/file.py`

## 3. Results and artefacts

The important section. For every result:

- **What was measured**, with the actual numbers. AUROC values, success rates,
  per-layer figures, sample sizes. Not "the probe performed reasonably".
- **Artefact path** for any plot or table (`results/auroc_heatmap.png`).
- **Config that produced it** — layers, seeds, episode count, hyperparameters.
- **Whether it is trustworthy yet.** Flag anything from a partial run, a single
  seed, or a pipeline that has not been sanity-checked.

Negative results belong here in the same detail as positive ones. The
UMAP/t-SNE null result was one of the more useful findings of Semester 1.

## 4. Problems, failures and dead ends

What broke, what was tried, what fixed it, how much time it cost. Include
approaches abandoned and why — that reasoning is hard to reconstruct later and
it is what the Problems Encountered section is assessed on.

## 5. Environment and infrastructure

Dependency conflicts, instance rebuilds, storage, anything now scripted rather
than done by hand. Note explicitly if a setup step was performed manually and
is not yet reproducible.

## 6. Literature and external inputs

Papers, repos or docs read on this machine that changed the implementation, and
what specifically changed as a result. Skip anything read but not acted on.

## 7. Linear delta

Drives the board update. Issue IDs are `THE-N` — see the table below.

| Issue | Title | Status change | Date change | Note |
|---|---|---|---|---|
| THE-2 | Per-layer probe AUROC | In Progress -> Done | — | heatmap generated |

Also list work done that maps to **no existing issue** — those become new rows
on the Gantt chart.

## 8. Next steps

Concrete and ordered. The immediate next command or decision, not aspirations.

## 9. Open questions for Don / Jen

Anything genuinely blocked on a supervisor decision. Empty is a fine answer.

## 10. Raw appendix

Key commands run, error messages verbatim, config diffs, short output snippets.
No length limit — this is reference material, not narrative.
```

---

## Current Linear issues

Reference for section 7. May drift; flag anything that looks stale.

| ID | Title | Due | Milestone |
|---|---|---|---|
| THE-1 | Logbook entries (S2) | 20 Nov | Project Mgmt & Problem Solving |
| THE-2 | Per-layer linear probe AUROC analysis | 14 Aug | Probe AUROC Analysis Complete |
| THE-3 | Presentation development | 14 Aug | Presentation / Seminar |
| THE-4 | Review literature (S2) | 16 Oct | Draft Thesis to Supervisor |
| THE-5 | Thesis document | 06 Nov | Submit Thesis |
| THE-6 | Train on internal constraints | 04 Sep | Internal Constraints Trained |
| THE-7 | Set up observer VLM | 11 Sep | Observer VLM Operational |
| THE-8 | Create and test FSA | 25 Sep | FSA Created & Tested |
| THE-9 | Train on constraint subsets | 16 Oct | Constraint Ablation Complete |
| THE-10 | Poster | 20 Nov | Submit Draft Poster |
| THE-11 | Oral exam presentation practice | 20 Nov | Oral Exam & Poster |

## Semester 2 week mapping

Mid-semester break sits between Weeks 8 and 9, so calendar dates and week
numbers drift apart after September. Use this rather than counting weeks.

| Wk | Commencing | Wk | Commencing |
|---|---|---|---|
| 1 | Mon 03 Aug | 9 | Mon 05 Oct |
| 2 | Mon 10 Aug | 10 | Mon 12 Oct |
| 3 | Mon 17 Aug | 11 | Mon 19 Oct |
| 4 | Mon 24 Aug | 12 | Mon 26 Oct |
| 5 | Mon 31 Aug | 13 | Mon 02 Nov |
| 6 | Mon 07 Sep | Exam 1 | Mon 09 Nov |
| 7 | Mon 14 Sep | Exam 2 | Mon 16 Nov |
| 8 | Mon 21 Sep | | |

Hard deadlines: **thesis Fri 06 Nov**, **oral exam and poster Fri 20 Nov**.

---

## Rules

1. **Evidence over recollection.** Every claim traces to a commit, path or
   command output. If it cannot be traced, mark it `[unverified]`.
2. **Do not compress.** Word limits are applied downstream. Over-supply here.
3. **Report failure plainly.** A week lost to a dependency conflict is a real
   logbook entry. Do not dress it up as progress.
4. **Never invent results.** No plausible-sounding numbers, no assumed run
   completions. If a run did not finish, say it did not finish.
5. **Flag anything not reproducible.** If it only worked because of a manual
   step, that is a finding, not an implementation detail.
6. **Write plainly.** First person, direct. This becomes an engineering
   student's logbook, not a paper abstract.
