# AI4Scientists Paper Revision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Style policy for prose edits:** when writing or rewriting paragraphs in `main.tex`, dispatch through `/academic-research-skills:ars-revision` (for content alignment with reviewer feedback) layered with `/humanizer` (for tone consistency with the existing manuscript). Do not let either skill rewrite material that is not the explicit edit target of the current task.

**Goal:** Revise the camera-ready ICML 2026 AI4Science position paper to absorb both reviewers' substantive critiques while sharpening the case-study contribution. No rebuttal; manuscript improvement only.

**Architecture:** F2 (case-study-first) backbone + F3 (methodology) layered solid + small F1 cherry as future-research direction. Single-author. Section 4 restructured to "Lessons and Limitations" with three subsections. New Appendix A providing bug-level classification table to replace the synthesized 80/10/10 figures with row-level data. Sharpened L1/L2/L3 ablation framing in §3.2 that directly answers R2's retrieval-vs-agency question. Figure 2 caption updated for coherence.

**Tech Stack:** LaTeX (ICML 2026 style: `icml2026.sty`, `icml2026.bst`), BibTeX, `gh` CLI for PR metadata, `claude-memory:get-token-insights` skill for v0.1.0 session JSONL ingest, `pdfinfo` for page-count regression.

---

## Source materials

- **Manuscript:** `/Users/nguyenmn/clax/drafts/ai4scientists/main.tex` (249 lines, 8-page PDF; venue cap 8 pages excl. references and appendices)
- **Bibliography:** `/Users/nguyenmn/clax/drafts/ai4scientists/references.bib`
- **Reviews:** `/Users/nguyenmn/clax/drafts/ai4scientists/reviews/review1.md` (rating 6, confidence 4) and `reviews/review2.md` (rating 7, confidence 1)
- **Primary case-study record (CHANGELOG):** `/Users/nguyenmn/clax/.claude/worktrees/benchmark-clax-pt/CHANGELOG.md` (1755 lines; clax-pt PT entries dense in 2026-04-02 → 2026-04-12 range)
- **Figures:** `/Users/nguyenmn/clax/drafts/ai4scientists/figures/fig1_the_wall.pdf` (accuracy-wall plot; source `accuracy_wall.tex`), `fig2_bug_taxonomy.pdf` (bug taxonomy)
- **Repository for paper artifact:** https://github.com/MinhMPA/clax-pt
- **Cross-check primary-record analysis:** by parallel agent (results captured in grilling session; bug table reconstructed from CHANGELOG; 13/14/15 count range judgement; intervention-level attribution flagged as low-confidence from CHANGELOG alone)

## Glossary of grilling-session terms used below

- **L1 — process scaffolding:** advisor prompt with zero domain content (e.g., "step back, is the current architecture wrong?"). Tests the agent's capacity to do meta-reflection given an explicit reminder.
- **L2 — domain hint:** advisor prompt injecting a physics concept (e.g., "consider anisotropic damping"). The conceptual leap is supplied; the agent maps it to code.
- **L3 — navigational pointer:** advisor pointer to a specific code path. Zero physics content; just retrieval.
- **F1 / F2 / F3:** three framings discussed during grilling. F1 = theory-first (load-bearing "explanatory agency" claim). F2 = case-study-first (modest, evidence-bounded). F3 = methodology-first (supervision-protocol contribution). Chosen blend: F2 backbone + F3 + small F1 as future research.

## Data to collect during execution (resolve each `[TBD ...]` when encountered)

| Marker | Resolved by | Default if unresolvable |
|---|---|---|
| `[TBD PR#9 DATE]` | `gh pr view 9 --repo MinhMPA/clax-pt --json createdAt` (Task 0.2) | "approximately 2026-04-12" with a note |
| `[TBD V0.1.0 DAYCOUNT]` | Computed from PR#9 date minus project-start date (also derived from JSONLs or earliest commit on `clax/ept.py`) | `12` (active work days, as originally reported) |
| `[TBD SESSION COUNT]` | `claude-memory:get-token-insights` skill on the v0.1.0 dev machine (Task 0.3) | `57` (from original submission) with caveat |
| `[TBD STUCK COUNT]` | Date-filtered JSONL ingest + intervention heuristic (Task 0.3) | `33 of 57` (from original submission) with caveat |
| `[TBD TOTAL COST USD]` | Same JSONL ingest, summing per-model rates | "Costs were not aggregated at the time; we omit a precise figure." |
| `[TBD L1 ATTEMPTS]` | Search JSONL transcripts in the 2026-04-XX RSD-stuck window for prompts containing "step back", "reconsider architecture", "is the architecture", "different approach" | "at least one" |

---

## Phase 0: Setup and data collection (do these first, in parallel where possible)

### Task 0.1: Confirm clean working tree on `paper_drafts` branch and snapshot the pre-revision PDF

**Files:**
- Verify: `/Users/nguyenmn/clax/drafts/ai4scientists/main.tex`
- Create: `/Users/nguyenmn/clax/drafts/ai4scientists/main.pdf.before-revision` (snapshot)

- [ ] **Step 1: Verify branch and clean working tree**

```bash
git -C /Users/nguyenmn/clax status --short
git -C /Users/nguyenmn/clax branch --show-current
```

Expected: branch is `paper_drafts`. Untracked files (`reviews/`, `review_results/`, `docs/superpowers/`) are acceptable; modifications inside `drafts/ai4scientists/main.tex` should be zero before starting.

- [ ] **Step 2: Snapshot current PDF for visual diff**

```bash
cp /Users/nguyenmn/clax/drafts/ai4scientists/main.pdf \
   /Users/nguyenmn/clax/drafts/ai4scientists/main.pdf.before-revision
pdfinfo /Users/nguyenmn/clax/drafts/ai4scientists/main.pdf.before-revision | grep -E "Pages|Page size"
```

Expected: `Pages: 8`, `Page size: 612 x 792 pts (letter)`. Record this as the baseline; the final revised PDF body must remain at or below 8 numbered-section pages (acknowledgments + appendix + references can extend beyond).

- [ ] **Step 3: Commit the snapshot**

```bash
git -C /Users/nguyenmn/clax add drafts/ai4scientists/main.pdf.before-revision
git -C /Users/nguyenmn/clax commit -m "snapshot: pre-revision PDF for visual diff"
```

### Task 0.2: Collect PR#9 open date from GitHub

**Files:** no manuscript changes; output is appended to this plan file under "Resolved TBDs."

- [ ] **Step 1: Query gh for PR#9 metadata**

```bash
gh pr view 9 --repo MinhMPA/clax-pt --json createdAt,title,headRefName,baseRefName
```

Expected: JSON with `createdAt: "YYYY-MM-DDTHH:MM:SSZ"`. Record this as `[TBD PR#9 DATE]`. The UTC calendar date is the v0.1.0 window end.

- [ ] **Step 2: Identify the v0.1.0 start date**

The first PT-relevant commit on `clax/ept.py` in the benchmark-clax-pt worktree marks the start:

```bash
cd /Users/nguyenmn/clax/.claude/worktrees/benchmark-clax-pt
git log --reverse --format='%ai %s' -- clax/ept.py 2>/dev/null | head -5
```

The first commit in this list is the v0.1.0 start.

- [ ] **Step 3: Compute active-day count**

`[TBD V0.1.0 DAYCOUNT]` = the supervisor-reported 12 active work days. Verify by counting weekdays between the v0.1.0 start (Step 2) and PR#9 date (Step 1); if the count is materially different (e.g., 14 active days), use the calendar-derived number and explain in the appendix preamble.

- [ ] **Step 4: Record resolved values in this plan**

Append the following section to this plan file (do not commit yet):

```markdown
## Resolved TBDs (filled during execution)

- PR#9 createdAt: 2026-MM-DD
- v0.1.0 start (first clax/ept.py commit): 2026-MM-DD
- Active work days: NN
```

### Task 0.3: Pull session JSONL data from v0.1.0 dev machine

**Files:** no manuscript changes; output appended to the Resolved TBDs block created in Task 0.2.

- [ ] **Step 1: Identify the dev machine**

The v0.1.0 sessions ran on the user's daily-driver machine for the late-March / early-April 2026 window. This Mac has only post-bring-up sessions (36 sessions in 2026-04-15 → 2026-05-19). The dev machine is likely a Bridges-2 igpu node or another workstation. If running this task on a different machine, ssh to the dev machine first.

```bash
ssh <dev-machine>
ls ~/.claude/projects/ | grep -i clax
du -sh ~/.claude/projects/-*clax* 2>/dev/null | sort -h
```

Expected: one or more `-*clax*` directories. The clax-pt project directory will contain ~57 `.jsonl` files dated within the v0.1.0 window.

- [ ] **Step 2: Run the token-insights skill from the dev machine**

```bash
# From a Claude Code session on the dev machine:
/claude-memory:get-token-insights
```

Expected: per-project breakdown including session count, total tokens by model, and an HTML dashboard at `~/.claude-memory/dashboard.html`. Record:
- `[TBD SESSION COUNT]` = total sessions in the clax-pt project dir within the v0.1.0 window
- `[TBD TOTAL COST USD]` = sum of per-model costs over that window (Opus 4.x + Sonnet 4.x)

If the skill is unavailable, fall back to the awk recipe from the grilling session that the parallel agent provided (Anthropic prices: Opus $15 in / $75 out / $18.75 cache_create / $1.50 cache_read per MTok; Sonnet $3 in / $15 out / $3.75 cache_create / $0.30 cache_read per MTok). Verify prices against Anthropic's current price card before quoting any final figure.

- [ ] **Step 3: Compute the stuck-session count**

Run the intervention heuristic over the JSONL window. Replace `<DEV_USER>` and `<PR9_ISO_DATE>` with actual values:

```bash
count=0; intervention=0
for f in ~/.claude/projects/-*clax*/*.jsonl; do
  first_ts=$(jq -r '.timestamp // empty' "$f" 2>/dev/null | head -1)
  if [[ "$first_ts" > "2026-04-02" && "$first_ts" < "<PR9_ISO_DATE>" ]]; then
    count=$((count+1))
    if jq -r 'select(.message.role=="user") | .message.content |
              if type=="string" then . else (.[]|select(.type=="text")|.text) end' "$f" 2>/dev/null | \
       grep -qiE "anisotropic|sigmatot|fudge|architectur|class-pt|nonlinear_pt|reconsider|step back|limiting case|alpha=0"; then
      intervention=$((intervention+1))
    fi
  fi
done
echo "Total v0.1.0 sessions: $count"
echo "Intervention-flagged: $intervention"
echo "Stuck (count - intervention): $((count - intervention))"
```

Record `[TBD STUCK COUNT]` as the count of sessions during the architectural-wall window (which the user described as 33 in the original submission). This heuristic is approximate; flag any large divergence (>20%) from the original figure for the supervisor's manual review before publication.

- [ ] **Step 4: Confirm physicist-initiated α=0 probe**

Search JSONL transcripts in the 50-minute window of 2026-04-09 13:57–14:47 UTC for the probe origin:

```bash
for f in ~/.claude/projects/-*clax*/*.jsonl; do
  if jq -r 'select(.timestamp >= "2026-04-09T13:57" and .timestamp <= "2026-04-09T14:47") |
            select(.message.role=="user") |
            .message.content | if type=="string" then . else (.[]|select(.type=="text")|.text) end' "$f" 2>/dev/null | \
     grep -iE "alpha=0|alpha = 0|set.*to zero|limiting case|check.*physical|fudge"; then
    echo "=== match in $f ==="
  fi
done
```

Expected: at least one user-side message asking about α=0 or limiting cases. Outcome:
- **If matches found:** confirms physicist-initiated framing; no change needed (this is the default assumption per the original submission).
- **If no matches found:** the probe was agent-initiated. Update Task 2.4 (§4.1 P1 rewrite) to soften "the catch came from a limiting-case stress test [physicist-prompted]" to "the catch came from a limiting-case stress test that surfaced within the agent's own exploration, consistent with our standing 'no fudge factors' policy though we cannot independently attribute its origin." This is a 1-sentence diff to the §4.1 P1 paragraph.

- [ ] **Step 5: Record resolved values; no commit yet**

Append under "Resolved TBDs" in this plan:

```markdown
- Session count (verified): NN  (original submission: 57)
- Stuck-session count (verified): MM  (original submission: 33)
- Total inference cost (Opus + Sonnet): $X.YZ
- α=0 probe initiation: physicist-initiated  [or: agent-initiated, see §4.1 P1 update note]
- L1 phrasing attempts (matches in window): K
```

---

## Phase 1: Structural scaffolding

These tasks change document structure before content rewrites land, so subsequent edits target the right sections.

### Task 1.1: Replace title and de-anonymize author block

**Files:**
- Modify: `/Users/nguyenmn/clax/drafts/ai4scientists/main.tex:14, 21-44`

- [ ] **Step 1: Add `orcidlink` to preamble**

In `main.tex`, find line 14:

```latex
\usepackage{xspace}
```

Replace with:

```latex
\usepackage{xspace}
\usepackage{orcidlink}
```

- [ ] **Step 2: Change `\icmltitlerunning` and `\icmltitle`**

Find lines 21–27:

```latex
\icmltitlerunning{Physics Is All You Need}

\begin{document}

\twocolumn[
  \icmltitle{Physics Is All You Need: Lessons from Physicist-Supervised \\
    AI Development of Scientific Software}
```

Replace with:

```latex
\icmltitlerunning{Physics Is All You Need? A Case Study}

\begin{document}

\twocolumn[
  \icmltitle{Physics Is All You Need? A Case Study in Physicist-Supervised \\
    AI Development of Scientific Software}
```

Rationale: addresses R1's "title contradicts the argument" critique by converting a declaration into an investigation. "A Case Study" pre-discloses the N=1 scope at first glance, defanging R1's "anecdote in generalization clothing" read.

- [ ] **Step 3: Replace `\icmlauthorlist` and affiliations**

Find lines 31–37:

```latex
  \begin{icmlauthorlist}
    \icmlauthor{Minh Nguyen}{kavli}
    \icmlauthor{Siddharth Mishra-Sharma}{anthropic}
  \end{icmlauthorlist}

  \icmlaffiliation{kavli}{Kavli IPMU, University of Tokyo, Kashiwa, Japan}
  \icmlaffiliation{anthropic}{Anthropic, San Francisco, CA, USA}
```

Replace with:

```latex
  \begin{icmlauthorlist}
    \icmlauthor{Nhat-Minh Nguyen\,\orcidlink{0000-0002-2542-7233}}{kavli,cd3,icise}
  \end{icmlauthorlist}

  \icmlaffiliation{kavli}{Kavli IPMU (WPI), UTIAS, The University of Tokyo, 5-1-5 Kashiwanoha, Kashiwa, Chiba 277-8583, Japan}
  \icmlaffiliation{cd3}{Center for Data-Driven Discovery, Kavli IPMU (WPI), UTIAS, The University of Tokyo, Kashiwa, Chiba 277-8583, Japan}
  \icmlaffiliation{icise}{Institute For Interdisciplinary Research in Science and Education, ICISE, Quy Nhon, 55121, Vietnam}
```

- [ ] **Step 4: Update `\icmlcorrespondingauthor`**

Find line 39:

```latex
  \icmlcorrespondingauthor{Minh Nguyen}{nhat.minh.nguyen.111@gmail.com}
```

Replace with:

```latex
  \icmlcorrespondingauthor{Nhat-Minh Nguyen}{nhat.minh.nguyen@ipmu.jp}
```

- [ ] **Step 5: Compile and visually verify**

```bash
cd /Users/nguyenmn/clax/drafts/ai4scientists
pdflatex -interaction=nonstopmode main.tex 2>&1 | tail -20
bibtex main 2>&1 | tail -5
pdflatex -interaction=nonstopmode main.tex 2>&1 | tail -5
pdflatex -interaction=nonstopmode main.tex 2>&1 | tail -5
pdfinfo main.pdf | grep -E "Pages|Page size"
```

Expected: compiles without fatal errors. Title shows the question-mark form, the author block shows only Nhat-Minh Nguyen with three affiliations and ORCID, the corresponding-author line shows the IPMU email. Page count unchanged or ±1.

If `orcidlink` is unavailable in the local LaTeX install, fall back to a manual ORCID footnote and remove the `\usepackage{orcidlink}` line.

- [ ] **Step 6: Commit**

```bash
git -C /Users/nguyenmn/clax add drafts/ai4scientists/main.tex drafts/ai4scientists/main.pdf
git -C /Users/nguyenmn/clax commit -m "paper: question-mark title and single-author de-anonymization"
```

### Task 1.2: Add Acknowledgments section before References

**Files:**
- Modify: `/Users/nguyenmn/clax/drafts/ai4scientists/main.tex` (immediately before `\bibliography{references}`)

- [ ] **Step 1: Insert acknowledgments**

In `main.tex`, find the line `\bibliography{references}` (around line 245). Insert immediately before it:

```latex
\section*{Acknowledgments}

I am grateful to Siddharth Mishra-Sharma for sharing about his work and for his collaboration on \texttt{clax}, the building block of \claxpt. I thank Ben Horowitz and Kazuyuki Akitsu for helpful discussions. I acknowledge support from the Japan Foundation for Promotion of Astronomy Research Grant and the JSPS KAKENHI Grant Numbers 25K23373 and 26H00404. This work was supported by World Premier International Research Center Initiative (WPI Initiative), MEXT, Japan.

```

- [ ] **Step 2: Compile and verify**

```bash
cd /Users/nguyenmn/clax/drafts/ai4scientists
pdflatex -interaction=nonstopmode main.tex 2>&1 | tail -5
pdflatex -interaction=nonstopmode main.tex 2>&1 | tail -5
pdftotext main.pdf - | grep -A1 "Acknowledgments"
```

Expected: section heading "Acknowledgments" renders before References with the supplied prose.

- [ ] **Step 3: Commit**

```bash
git -C /Users/nguyenmn/clax add drafts/ai4scientists/main.tex drafts/ai4scientists/main.pdf
git -C /Users/nguyenmn/clax commit -m "paper: add Acknowledgments section"
```

### Task 1.3: Restructure §4 and §5 into "Lessons and Limitations" with three subsections; renumber downstream

**Files:**
- Modify: `/Users/nguyenmn/clax/drafts/ai4scientists/main.tex:172-223`

- [ ] **Step 1: Identify the exact text spans**

In `main.tex`:
- Lines 172–198 hold current §4 "Three Principles" (heading + intro paragraph + P1/P2/P3 paragraphs).
- Lines 200–212 hold current §5 "Implications and Limitations" intro paragraph (contains the 80/10/10 prose, three counterfactuals, and the remediation-paths paragraph).
- Lines 214–223 hold current §5.1 "Credit, Attribution, and Governance" subsection.

The next section header is `\section{Related Work}` (currently §6; will become §5 after renumbering — `cleveref` handles the cross-references automatically).

- [ ] **Step 2: Replace the entire 172–223 block with the new structure**

Read the current content of lines 172–223 first; preserve substantive paragraphs (P1, P2, P3, governance prose). Headings and intro paragraphs change.

Replace lines 172–223 with:

```latex
% ===========================================================================
\section{Lessons and Limitations}
\label{sec:lessons}

The case study suggests three patterns observed in this project, each grounded in a specific failure mode we encountered. We organize them around the failure mode and the supervision practice that emerged in response; we do not claim these are universal principles.

\subsection{Managing Autonomy in Scientific Software}
\label{sec:managing}

\paragraph{P1: Oracle testing verifies what, not why.}

[PRESERVE the existing P1 paragraph from current lines 178-184 verbatim, EXCEPT delete the "subtler instance" Boltzmann sentences (Task 2.8) and rewrite the "Had multi-cosmology testing been enforced from day one" counterfactual (Task 2.4). The body of P1 will be edited by those tasks.]

\paragraph{P2: Shared memory prevents re-exploration but not architectural loops.}

[PRESERVE the existing P2 paragraph from current lines 186-190 verbatim. Task 5.4 may scope-bind "approximately 5--10 sessions" framing if needed.]

\paragraph{P3: The irreducible human role is architectural and physical judgment.}

[PRESERVE the existing P3 paragraph from current lines 192-198 verbatim. Task 5.4 will scope-bind any over-generalizing phrases.]

\subsection{Credit, Attribution, and Responsibility}
\label{sec:credit}

[PRESERVE the existing §5.1 subsection from current lines 215-223, with the heading already changed in this step. Task 2.5 will trim to one paragraph and replace "governance"-flavored phrasing.]

\subsection{What This Case Study Cannot Show}
\label{sec:cannot}

[PLACEHOLDER: body written in Task 2.6.]

```

Also delete current lines 200–212 in their entirety (the former §5 intro paragraph). Its content is redistributed:
- 80/10/10 prose → cut entirely, replaced by Appendix A in Phase 4.
- The three counterfactual sentences (no-fudge-factor, multi-cosmo, session-count escalation) → the multi-cosmo one is rewritten in Task 2.4 into §4.1 P1; the other two are absorbed into §4.1 P2 (session-count escalation already belongs there) or cut.
- The "two concrete remediation paths" paragraph (retrieval-augmented reasoning + physics-audit prompting) → absorbed into §4.3 by Task 2.6.

To enable Task 2.4 and Task 2.6 to find the source material, before deleting current lines 200–212, copy the multi-cosmo sentence and the remediation paragraph into a comment block at the very end of `main.tex` (after `\end{document}`):

```latex
% =====================================================================
% MATERIAL EXTRACTED FOR REDISTRIBUTION (delete after Phase 2 complete)
% =====================================================================
% From former §5 intro:
%
% Had multi-cosmology testing been enforced from day one rather than added
% after the fudge factor was discovered, the calibration would have been
% caught sooner. alpha=0.27 is fit to fiducial Planck 2018 parameters and
% would produce visibly wrong BAO amplitudes at omega_b +/- 20%.
%
% Looking forward, two concrete remediation paths could narrow the gap
% between autonomous and human-required bugs without requiring qualitative
% advances in reasoning. First, retrieval-augmented reasoning over the full
% reference codebase, including code paths the agent did not initially
% consult, could surface relevant physics without human guidance...
% [paste the rest of the remediation paragraph here verbatim]
```

Task 2.4 and Task 2.6 will reference this comment block and remove it once their rewrites land.

- [ ] **Step 3: Update `\cref` label references throughout the document**

The label `sec:principles` no longer exists; `sec:implications` no longer exists; `sec:governance` is renamed to `sec:credit`. Find affected references:

```bash
grep -n "sec:principles\|sec:implications\|sec:governance" /Users/nguyenmn/clax/drafts/ai4scientists/main.tex
```

For each match, apply the mapping:
- `sec:principles` → `sec:lessons` or `sec:managing` (context-dependent; default to `sec:managing` for references to the three patterns)
- `sec:implications` → `sec:lessons` (the new umbrella) or `sec:cannot` (if the reference is specifically to the limitations content)
- `sec:governance` → `sec:credit`

- [ ] **Step 4: Compile and confirm structure**

```bash
cd /Users/nguyenmn/clax/drafts/ai4scientists
pdflatex -interaction=nonstopmode main.tex 2>&1 | tail -10
pdflatex -interaction=nonstopmode main.tex 2>&1 | tail -5
pdftotext -layout main.pdf - | grep -E "^[0-9]+\.[ ]|^[0-9]+\.[0-9]" | head -15
pdfinfo main.pdf | grep Pages
```

Expected TOC structure (extracted from PDF): §1 Introduction, §2 The Project, §3 Bug Taxonomy, §4 Lessons and Limitations (with subsections 4.1, 4.2, 4.3), §5 Related Work, §6 Conclusion. §4.3 body is the placeholder bracket at this stage.

- [ ] **Step 5: Commit**

```bash
git -C /Users/nguyenmn/clax add drafts/ai4scientists/main.tex drafts/ai4scientists/main.pdf
git -C /Users/nguyenmn/clax commit -m "paper: restructure to Lessons and Limitations (S2 layout)"
```

### Task 1.4: Create Appendix A scaffold

**Files:**
- Modify: `/Users/nguyenmn/clax/drafts/ai4scientists/main.tex` (after Acknowledgments, before `\bibliography`)

- [ ] **Step 1: Insert appendix scaffold**

Find the `\section*{Acknowledgments}` block added in Task 1.2. After the acknowledgments paragraph (and before `\bibliography{references}`), insert:

```latex
\appendix

\section{Issue-Level Classification}
\label{app:bug-table}

[PLACEHOLDER: populated in Task 4.1.]

```

`\appendix` resets section numbering to letters (A, B, C...).

- [ ] **Step 2: Compile and verify**

```bash
cd /Users/nguyenmn/clax/drafts/ai4scientists
pdflatex -interaction=nonstopmode main.tex 2>&1 | tail -5
pdflatex -interaction=nonstopmode main.tex 2>&1 | tail -5
pdftotext main.pdf - | grep -E "^A\.[ ]+Issue|^Appendix A"
```

Expected: "A. Issue-Level Classification" appears after Acknowledgments. Body is the placeholder at this stage.

- [ ] **Step 3: Commit**

```bash
git -C /Users/nguyenmn/clax add drafts/ai4scientists/main.tex drafts/ai4scientists/main.pdf
git -C /Users/nguyenmn/clax commit -m "paper: Appendix A scaffold (issue-level classification)"
```

---

## Phase 2: Main-text content rewrites

> **Tone reminder:** prose written in Phase 2 should match the existing manuscript's voice — measured, technical, present tense, sentence-level claims grounded in evidence. Use `/academic-research-skills:ars-revision` + `/humanizer` to align tone when generating new paragraphs; preserve the manuscript's existing register (no buzzwords, no overclaim, no AI-flavored hedging).

### Task 2.1: Rewrite §3.2 around L1/L2/L3 ablation framing with factual correction

**Files:**
- Modify: `/Users/nguyenmn/clax/drafts/ai4scientists/main.tex:154-156` (current §3.2 "Why could the agent not find this itself?" paragraph)

- [ ] **Step 1: Locate the paragraph**

Find the paragraph in §3.2 starting with `Why could the agent not find this itself? Two factors conspired.`. This paragraph contains the factually incorrect claim that "the agent consulted only the simpler [path]."

- [ ] **Step 2: Replace the paragraph**

Find the exact text:

```latex
Why could the agent not find this itself? Two factors conspired. First, \classpt has two parallel code paths for these integrals. The agent consulted only the simpler one, which uses isotropic damping and analytic projections---exactly the architecture the agent had implemented. The correct anisotropic formula lives in the other path, which handles geometric distortion corrections that the test configuration did not require. Second, recognizing that the architecture is wrong (rather than merely mis-parameterized) requires understanding the physics of anisotropic BAO damping: specifically, that the $\mu$-dependence of $\Sigma^2_\mathrm{tot}$ makes the exponential non-polynomial in $\mu$. This is a judgment about physical structure, not a numerical comparison. Of these two factors, the first is an engineering limitation (incomplete codebase search) addressable by retrieval tools. The second is the explanatory agency gap that no amount of search can close.
```

Replace with:

```latex
Why could the agent not find this itself? The agent had autonomously surveyed both \classpt code paths during initial exploration: the simpler path (isotropic damping with analytic Legendre projections) and the more complex path (anisotropic damping handling geometric distortion corrections). It selected the simpler path as its implementation target---a reasonable choice given the test configuration did not require geometric distortion---and proceeded to implement it. What it could not do, across 33 sessions of unsuccessful coefficient adjustment, was \emph{re-evaluate} that selection: ask whether the path it had not chosen might be the relevant one for the failing tests. The physicist explicitly attempted process scaffolding (a generic, domain-free reconsider-the-architecture prompt: ``the current architecture may be the wrong frame; please reconsider whether your existing kernel-matrix structure can represent the target physics, rather than tuning coefficients within it''); the agent reaffirmed its design and continued coefficient adjustments. The architectural redesign was triggered only when the physicist supplied the relevant physics concept (anisotropic BAO damping in redshift space). Given that concept, the agent immediately recognized the previously-surveyed second branch as the appropriate implementation target. Codebase retrieval was already complete in this case, so we did not run a controlled ablation isolating the conceptual injection from a hypothetical navigational-pointer-only intervention. We therefore cannot rule out that aggressive retrieval over the full \classpt codebase, delivered to a different agent without our reconsider-the-architecture prompt, might have surfaced the anisotropic branch unprompted. What we can say is that within this case, the gap was in generating the physics question that selects between mapped alternatives, not in scanning the codebase.
```

Note: this rewrite uses the L1 / L2 / L3 *content* without committing to the L1/L2/L3 *vocabulary* in §3.2's body (the vocabulary is too heavy for a single occurrence; §4.3 uses the L1/L2/L3 labels once explicitly).

- [ ] **Step 3: Compile and verify**

```bash
cd /Users/nguyenmn/clax/drafts/ai4scientists
pdflatex -interaction=nonstopmode main.tex 2>&1 | tail -5
pdflatex -interaction=nonstopmode main.tex 2>&1 | tail -5
pdfinfo main.pdf | grep Pages
```

Expected: paragraph reads cleanly; the factually-wrong "consulted only the simpler one" claim is gone; the architectural retrieval-vs-agency distinction is honest.

- [ ] **Step 4: Commit**

```bash
git -C /Users/nguyenmn/clax add drafts/ai4scientists/main.tex drafts/ai4scientists/main.pdf
git -C /Users/nguyenmn/clax commit -m "paper: §3.2 L1/L2 ablation framing and factual correction"
```

### Task 2.2: Add positive-autonomy paragraph (CLASS-PT codebase mapping)

**Files:**
- Modify: `/Users/nguyenmn/clax/drafts/ai4scientists/main.tex` (immediately after §3.1's closing paragraph, before §3.2's subsection header)

- [ ] **Step 1: Locate the §3.1 closing paragraph**

Find the line that closes §3.1 "Autonomous and Human-Accelerated Issues":

```latex
In all these cases, the pattern was the same: the oracle test produced a clear numerical discrepancy, the agent compared intermediate quantities against reference data to localize the error, and the fix was a direct transcription correction.
```

- [ ] **Step 2: Insert the new paragraph after it**

Append (before the `\subsection{Case Study: The Accuracy Wall}` line):

```latex
Beyond bug-fixing, the agent autonomously performed substantive codebase archaeology that the supervisor had not directed. During the first sessions of bring-up, the agent independently mapped the structure of the \classpt reference source---identifying that the EFT calculation lives in two parallel code paths with different treatments of the redshift-space integrals---without being told to look for this distinction and without being given a guide to the source layout. This survey was complete enough that, once the physicist later supplied the conceptual hint about anisotropic damping (\cref{sec:wall}), the agent could re-target its implementation to the previously-surveyed second branch without further retrieval. The codebase scan, then, was not a limitation in this case: the limitation was elsewhere, in the agent's inability to re-evaluate which of the two mapped branches its failing tests implicated.
```

- [ ] **Step 3: Compile and verify**

```bash
cd /Users/nguyenmn/clax/drafts/ai4scientists
pdflatex -interaction=nonstopmode main.tex 2>&1 | tail -5
pdflatex -interaction=nonstopmode main.tex 2>&1 | tail -5
```

Expected: new paragraph sits between §3.1 closing and §3.2 header. The cross-reference `\cref{sec:wall}` resolves correctly to §3.2.

- [ ] **Step 4: Commit**

```bash
git -C /Users/nguyenmn/clax add drafts/ai4scientists/main.tex drafts/ai4scientists/main.pdf
git -C /Users/nguyenmn/clax commit -m "paper: add positive-autonomy paragraph (CLASS-PT codebase mapping)"
```

### Task 2.3: Add 50-minute fudge-factor introduction→removal window to §3.3

**Files:**
- Modify: `/Users/nguyenmn/clax/drafts/ai4scientists/main.tex` (§3.3 "Case Study: The Fudge Factor," near the commit-and-passing-tests sentence)

- [ ] **Step 1: Locate the commit-and-passing sentence**

Find in §3.3:

```latex
The agent committed this fix with a passing test suite.
```

- [ ] **Step 2: Replace with timestamped version**

Replace with:

```latex
The agent committed this fix with a passing test suite at 13:57 on 2026-04-09. Fifty minutes later, at 14:47 in the same session, the agent committed its replacement: a re-derivation from the \classpt C source (\texttt{nonlinear\_pt.c} line 9388) that moved the tree-level computation inside the existing Gauss--Legendre loop with anisotropic damping $\Sigma^2_\mathrm{tot}(\mu)$, eliminating $\alpha$ entirely.
```

This anchors the catch timeline in verifiable git commit timestamps and concretizes the "fast catch" detail Task 2.4 builds on.

- [ ] **Step 3: Compile and verify**

```bash
cd /Users/nguyenmn/clax/drafts/ai4scientists
pdflatex -interaction=nonstopmode main.tex 2>&1 | tail -5
pdflatex -interaction=nonstopmode main.tex 2>&1 | tail -5
```

- [ ] **Step 4: Commit**

```bash
git -C /Users/nguyenmn/clax add drafts/ai4scientists/main.tex drafts/ai4scientists/main.pdf
git -C /Users/nguyenmn/clax commit -m "paper: §3.3 add 50-minute fudge-factor introduction→removal window"
```

### Task 2.4: Rewrite the multi-cosmo counterfactual inside §4.1 P1

**Files:**
- Modify: `/Users/nguyenmn/clax/drafts/ai4scientists/main.tex` (§4.1 P1 paragraph)

- [ ] **Step 1: Locate the P1 closing sentence**

In §4.1 P1 (which originates at current lines 180–184), find the closing sentence:

```latex
The defense we found effective was twofold: test at diverse parameter points beyond the fiducial calibration (so that single-point calibrations are exposed when parameters shift), and test structural properties rather than just numerical values.
```

- [ ] **Step 2: Replace with the corrected counterfactual**

Replace that single sentence with:

```latex
Multi-cosmology testing was active from project inception, but it was not what caught $\alpha=0.27$. The catch came from a limiting-case stress test: setting the tuned coefficient to $\alpha=0$ and observing that real-space errors exceeded 1\%, which exposed the calibration as compensating for a structural defect rather than encoding a physical correction. The fix was a re-derivation from the \classpt C source (\texttt{nonlinear\_pt.c} line 9388), implementing anisotropic damping $\Sigma^2_\mathrm{tot}(\mu)$ inside the existing Gauss--Legendre loop and dropping $\alpha$ entirely (\cref{sec:fudge}). The 50 minutes between commits is evidence that the supervision practice---operationalizing ``no fudge factors'' into a parameter-boundary probe---fired efficiently in this instance, but as an ad-hoc check rather than as an automated pre-commit gate. Two complementary defenses follow: test at diverse parameter points beyond the fiducial calibration so single-point calibrations are exposed when parameters shift, and automate limiting-case probes (set each tuned coefficient to a boundary value, re-run the oracle) as a mandatory pre-commit step so the fudge-rejection mechanism does not depend on a supervisor noticing.
```

- [ ] **Step 3: Compile and verify**

```bash
cd /Users/nguyenmn/clax/drafts/ai4scientists
pdflatex -interaction=nonstopmode main.tex 2>&1 | tail -5
pdflatex -interaction=nonstopmode main.tex 2>&1 | tail -5
```

- [ ] **Step 4: Commit**

```bash
git -C /Users/nguyenmn/clax add drafts/ai4scientists/main.tex drafts/ai4scientists/main.pdf
git -C /Users/nguyenmn/clax commit -m "paper: §4.1 P1 multi-cosmo counterfactual → α=0 probe mechanism"
```

### Task 2.5: Trim §4.2 to one paragraph; replace "governance" with "responsibility"

**Files:**
- Modify: `/Users/nguyenmn/clax/drafts/ai4scientists/main.tex` (§4.2 Credit, Attribution, and Responsibility)

- [ ] **Step 1: Confirm subsection header was renamed in Task 1.3**

```bash
grep -n "subsection{Credit" /Users/nguyenmn/clax/drafts/ai4scientists/main.tex
```

Expected: shows `\subsection{Credit, Attribution, and Responsibility}` once. If the heading still says "Governance," fix it now.

- [ ] **Step 2: Replace the §4.2 body**

The current §4.2 has two long paragraphs (~200 words across). Replace the entire body with the single paragraph:

```latex
The division of labor raises an unavoidable question about credit. The agent performed the bulk of session-time (implementation, debugging, transcription); the physicist performed a small fraction but supplied $100\%$ of the decisive architectural and physical judgments. Contribution weight should reflect \emph{irreplaceability}, not volume: the three load-bearing interventions (the architectural redesign, the rejection of the calibration patch, and the identification of the correct anisotropic formula) were not substitutable by any agent in this study, while the autonomous bug fixes could have been performed by any sufficiently capable coding agent against the same oracle. The analogy to graduate advising is instructive: the physicist's interventions resembled advising a student. What distinguishes the agent from a graduate student in this work is not the volume of code produced but the absence of \emph{explanatory agency}---the capacity to spontaneously generate questions about whether the current solution frame is correct, and to defend the answers under scrutiny. A student who discovered $\alpha = 0.27$ would acquire discomfort with an unphysical parameter and eventually formulate the diagnostic question without prompting; the agent could not. Until agents develop this capacity, authorship and intellectual responsibility for AI-assisted scientific software should remain with the supervising human, and AI-assisted scientific software should ship with a supervision log as provenance documentation, analogous to a lab notebook. The supervision log for this work is publicly available alongside the code (\cref{sec:availability}).
```

- [ ] **Step 3: Compile and verify**

```bash
cd /Users/nguyenmn/clax/drafts/ai4scientists
pdflatex -interaction=nonstopmode main.tex 2>&1 | tail -5
pdflatex -interaction=nonstopmode main.tex 2>&1 | tail -5
grep -c -i "governance" /Users/nguyenmn/clax/drafts/ai4scientists/main.tex
```

Expected: `governance` count is zero or 1 (one acceptable in a citation context only; cut if present in prose).

- [ ] **Step 4: Commit**

```bash
git -C /Users/nguyenmn/clax add drafts/ai4scientists/main.tex drafts/ai4scientists/main.pdf
git -C /Users/nguyenmn/clax commit -m "paper: §4.2 trim and shift to responsibility framing"
```

### Task 2.6: Write §4.3 "What This Case Study Cannot Show"

**Files:**
- Modify: `/Users/nguyenmn/clax/drafts/ai4scientists/main.tex` (§4.3 body placeholder from Task 1.3)

- [ ] **Step 1: Replace the placeholder**

Find `[PLACEHOLDER: body written in Task 2.6.]` inside `\subsection{What This Case Study Cannot Show}`. Replace with:

```latex
This case study is bounded in four specific ways readers should hold against any generalization.

\paragraph{Counterexamples not observed.}
The supervisor adopted a hands-off approach, intervening only after multiple sessions without progress; the agent reviewed and confirmed which dead ends were its own explorations. We observed no cases in which the supervisor's intuition was wrong, misleading, or less efficient than the agent's autonomous trajectory. This is itself a selection bias: the supervisor intervened when the agent was stuck, not when it was on-track. A symmetric study---one in which the agent's autonomous decisions are independently judged against the supervisor's interventions---would be required to characterize the relative error rates of human and agent in this domain. [TBD: insert one sentence on inference cost here per Task 5.1.]

\paragraph{Retrieval-vs-agency limitation.}
The architectural diagnosis (\cref{sec:wall}) was triggered by injecting a physics concept (anisotropic damping). We did not run a controlled ablation isolating the conceptual injection from a hypothetical navigational pointer with no physics content. The agent's codebase mapping was already complete in this case (\cref{sec:bugs}), so the relevant retrieval was already done---but we cannot rule out that a sufficiently aggressive retrieval system, delivered to a different agent that had not yet built our agent's prior survey, might have surfaced the anisotropic branch unprompted. Controlled ablations across multiple stuck-agent cases---separating process scaffolding (a generic reconsider-the-architecture prompt with no physics content), conceptual injection (a physics concept with no code pointer), and retrieval augmentation (a code pointer with no physics content), with retrieval state held constant---are a natural follow-up.

\paragraph{Single supervisor, single domain.}
The case study and the post-hoc classification rest on one supervising physicist and one domain (cosmological perturbation theory). The intervention-level classification was reviewed by the agent, which confirmed which dead ends were its own work; this is a documented second-pass check, not an independent human inter-rater study. Row-level bug data with confidence flags and a separate independent reconstruction from the development log is provided in Appendix~\ref{app:bug-table}; readers can disagree with individual rows.

\paragraph{Two concrete remediation paths.}
Two interventions, complementary to scaling, could narrow the gap between autonomous and human-required bugs observed here. Retrieval-augmented reasoning over the full reference codebase could in principle surface relevant physics without human guidance, though as noted above retrieval was already complete in our case. Explicit ``physics audit'' prompting---systematically asking the agent whether every tuned parameter corresponds to a physical quantity in the reference theory---would operationalize the diagnostic question that caught the fudge factor. The physicist's intervention reduced to a single query (``does $\alpha=0.27$ appear anywhere in \classpt?''), which the agent answered correctly once asked but could not generate unprompted. Embedding such queries as mandatory checkpoints after any parameter introduction would convert an ad hoc human insight into a reproducible protocol step. The deeper open question---whether agents with greater capacity than current LLMs would spontaneously generate the diagnostic question without prompting---is the future-research direction this case study suggests but cannot itself answer.
```

- [ ] **Step 2: Remove the extraction comment block at the end of `main.tex`**

If Task 1.3 added the temporary `% MATERIAL EXTRACTED FOR REDISTRIBUTION` comment block after `\end{document}`, delete it now. The remediation paragraph has been absorbed.

- [ ] **Step 3: Compile and verify**

```bash
cd /Users/nguyenmn/clax/drafts/ai4scientists
pdflatex -interaction=nonstopmode main.tex 2>&1 | tail -5
pdflatex -interaction=nonstopmode main.tex 2>&1 | tail -5
pdfinfo main.pdf | grep Pages
```

Expected: §4.3 has four `\paragraph{}` blocks. The remaining `[TBD]` (cost-sentence) will be resolved in Task 5.1.

- [ ] **Step 4: Commit**

```bash
git -C /Users/nguyenmn/clax add drafts/ai4scientists/main.tex drafts/ai4scientists/main.pdf
git -C /Users/nguyenmn/clax commit -m "paper: §4.3 What This Case Study Cannot Show"
```

### Task 2.7: Sweep up former §5 remnants

**Files:**
- Modify: `/Users/nguyenmn/clax/drafts/ai4scientists/main.tex`

- [ ] **Step 1: Search for residual 80/10/10 prose**

```bash
grep -n "approximately 80\|approximately 10\|80\%\|80/10/10" /Users/nguyenmn/clax/drafts/ai4scientists/main.tex
```

If matches exist, delete the entire sentences containing them. The percentages are replaced by Appendix A.

- [ ] **Step 2: Search for residual counterfactual phrasings**

```bash
grep -n "Had the\|Had multi-cosmology\|Had the session-count" /Users/nguyenmn/clax/drafts/ai4scientists/main.tex
```

Expected: zero matches in this exact form (the multi-cosmo one was rewritten by Task 2.4; the no-fudge-factor and session-count ones should have been removed by Task 1.3). If any remain, delete them.

- [ ] **Step 3: Compile and verify**

```bash
cd /Users/nguyenmn/clax/drafts/ai4scientists
pdflatex -interaction=nonstopmode main.tex 2>&1 | tail -5
pdflatex -interaction=nonstopmode main.tex 2>&1 | tail -5
```

- [ ] **Step 4: Commit (only if diff exists)**

```bash
git -C /Users/nguyenmn/clax diff --quiet drafts/ai4scientists/main.tex || {
  git -C /Users/nguyenmn/clax add drafts/ai4scientists/main.tex drafts/ai4scientists/main.pdf
  git -C /Users/nguyenmn/clax commit -m "paper: scrub former §5 remnants"
}
```

### Task 2.8: Cut the Boltzmann/clax typo passage from §4.1 P1

**Files:**
- Modify: `/Users/nguyenmn/clax/drafts/ai4scientists/main.tex` (§4.1 P1)

- [ ] **Step 1: Locate the passage**

Find in §4.1 P1:

```latex
A subtler instance appeared during the upstream Boltzmann solver development. The agent misdiagnosed a proton-to-hydrogen mass ratio typo ($m_p/m_H = 1.0$ vs.\ the correct 0.9994) as ``recombination solver bias'' and committed compensating changes that passed all tests for months before the one-character error was found.
```

- [ ] **Step 2: Delete both sentences**

Remove the two sentences. Rationale: redundant with the fudge-factor example (both illustrate "oracle testing verifies what, not why"); introduces clax/clax-pt scope confusion; the "for months" framing was imprecise; the F2 case-study framing benefits from rigorous in-domain scope.

After deletion, the surrounding P1 paragraph should still read fluidly; verify.

- [ ] **Step 3: Compile and verify**

```bash
cd /Users/nguyenmn/clax/drafts/ai4scientists
pdflatex -interaction=nonstopmode main.tex 2>&1 | tail -5
pdflatex -interaction=nonstopmode main.tex 2>&1 | tail -5
grep -c "Boltzmann solver development\|m_p/m_H\|recombination solver bias" /Users/nguyenmn/clax/drafts/ai4scientists/main.tex
```

Expected: zero matches.

- [ ] **Step 4: Commit**

```bash
git -C /Users/nguyenmn/clax add drafts/ai4scientists/main.tex drafts/ai4scientists/main.pdf
git -C /Users/nguyenmn/clax commit -m "paper: cut Boltzmann/clax typo passage (out of scope, redundant)"
```

### Task 2.9: Rewrite Conclusion

**Files:**
- Modify: `/Users/nguyenmn/clax/drafts/ai4scientists/main.tex` (current §7 Conclusion paragraph, around lines 240–243)

- [ ] **Step 1: Locate the Conclusion**

Find:

```latex
In our case study, the AI agent functioned as a highly capable tool---not a co-author and certainly not an autonomous scientist. It could implement, debug, and validate scientific software at speed and scale impractical for a solo physicist, but it lacked the explanatory agency to judge whether its solutions were physically meaningful---whether they produced right numbers for the right reasons. The supervision protocol (oracle testing, shared memory, the no-fudge-factor rule, and the session-count escalation trigger) was the mechanism that transformed raw agent output into trustworthy scientific code. Improving this protocol, rather than solely improving model capability, is the more direct path to reliable AI-assisted scientific software development. The full supervision log, CHANGELOG, and commit history will be made publicly available.
```

- [ ] **Step 2: Replace**

```latex
In this case study, the AI agent functioned as a highly capable tool: it could implement, debug, and validate scientific software at speed and scale impractical for a solo physicist, but it lacked the capacity to spontaneously generate the meta-question of whether its solution frame was correct. The supervision practices we used---oracle testing against an established reference, shared session memory, the explicit ``no fudge factors'' rule operationalized into a limiting-case parameter probe, and supervisor-led escalation when sessions stalled---were the mechanism that transformed raw agent output into code we were willing to call trustworthy. Whether such practices generalize beyond this one project, this one supervisor, and this one domain, and whether future agents with greater capacity than current LLMs will spontaneously generate the diagnostic question without prompting, are questions our evidence cannot settle. The full supervision log, CHANGELOG, and commit history are available alongside the code (\cref{sec:availability}).
```

- [ ] **Step 3: Compile and verify**

```bash
cd /Users/nguyenmn/clax/drafts/ai4scientists
pdflatex -interaction=nonstopmode main.tex 2>&1 | tail -5
pdflatex -interaction=nonstopmode main.tex 2>&1 | tail -5
```

- [ ] **Step 4: Commit**

```bash
git -C /Users/nguyenmn/clax add drafts/ai4scientists/main.tex drafts/ai4scientists/main.pdf
git -C /Users/nguyenmn/clax commit -m "paper: §6 Conclusion rewrite (scope-bound, drop overclaim)"
```

---

## Phase 3: Number and label hygiene

### Task 3.1: Rewrite the Abstract

**Files:**
- Modify: `/Users/nguyenmn/clax/drafts/ai4scientists/main.tex:49-55`

- [ ] **Step 1: Locate the abstract**

Lines 49–55 contain the current abstract.

- [ ] **Step 2: Replace**

```latex
\begin{abstract}
We present a quantified case study of one physicist supervising an AI coding agent (Claude Code, Sonnet and Opus models) over [TBD V0.1.0 DAYCOUNT] active work days and [TBD SESSION COUNT] sessions to build a differentiable one-loop perturbation theory module in JAX (\claxpt, ${\sim}2{,}100$ lines), validated to ${\lesssim}1\%$ accuracy against the established Fortran reference \classpt. We documented 15 distinct supervision events during the v0.1.0 development window and classified each by intervention level (Appendix~\ref{app:bug-table}).

Ten were resolved autonomously by the agent iterating against oracle tests; two more were accelerated by the physicist spotting magnitude discrepancies invisible to shape-based comparisons; three required essential human physics judgment. The agent spent [TBD STUCK COUNT] of the [TBD SESSION COUNT] sessions adjusting coefficients within a code architecture that could not represent the target physics, and could not re-evaluate its choice of \classpt branch even when the physicist explicitly prompted reconsideration; only an injected physics concept (anisotropic BAO damping) triggered the redesign. Separately, the agent introduced a calibrated scalar correction that passed all oracle tests but corresponded to no quantity in the reference theory; the supervision practice operationalizing ``no fudge factors'' into a parameter-boundary probe caught and replaced it within 50 minutes.

In this case study, supervision design---not model capability---determined whether the agent's output was trustworthy. The strongest pattern we observed is a gap between the agent's capacity to optimize within a given solution frame and its capacity to spontaneously question whether the frame itself is correct; controlled ablations across multiple stuck-agent cases would be needed to characterize this gap precisely. Whether the practices we used generalize beyond this one project, supervisor, and domain is a question our evidence does not settle.
\end{abstract}
```

- [ ] **Step 3: Compile and verify**

```bash
cd /Users/nguyenmn/clax/drafts/ai4scientists
pdflatex -interaction=nonstopmode main.tex 2>&1 | tail -5
pdflatex -interaction=nonstopmode main.tex 2>&1 | tail -5
grep -c "TBD" /Users/nguyenmn/clax/drafts/ai4scientists/main.tex
```

Expected: TBD count is 4 (V0.1.0 DAYCOUNT, SESSION COUNT × 2, STUCK COUNT). These resolve in Task 5.1.

- [ ] **Step 4: Commit**

```bash
git -C /Users/nguyenmn/clax add drafts/ai4scientists/main.tex drafts/ai4scientists/main.pdf
git -C /Users/nguyenmn/clax commit -m "paper: rewrite abstract (scope-bound, appendix pointer, TBD markers)"
```

### Task 3.2: Final scrub of synthesized percentages

**Files:**
- Modify: `/Users/nguyenmn/clax/drafts/ai4scientists/main.tex`

- [ ] **Step 1: Grep for percentages**

```bash
grep -n "approximately 80\|approximately 10\|80\%\|80/10" /Users/nguyenmn/clax/drafts/ai4scientists/main.tex
```

- [ ] **Step 2: For each match, replace with anchor numbers or cut**

Acceptable anchor numbers: "10 of 15 issues," "[TBD STUCK COUNT] of [TBD SESSION COUNT] sessions." Cut all synthesized session-weighted percentages.

- [ ] **Step 3: Compile and verify**

```bash
cd /Users/nguyenmn/clax/drafts/ai4scientists
pdflatex -interaction=nonstopmode main.tex 2>&1 | tail -5
grep -c "approximately 80\|approximately 10\|80\%\|80/10" /Users/nguyenmn/clax/drafts/ai4scientists/main.tex
```

Expected: zero matches.

- [ ] **Step 4: Commit (only if diff exists)**

```bash
git -C /Users/nguyenmn/clax diff --quiet drafts/ai4scientists/main.tex || {
  git -C /Users/nguyenmn/clax add drafts/ai4scientists/main.tex drafts/ai4scientists/main.pdf
  git -C /Users/nguyenmn/clax commit -m "paper: final scrub of synthesized session-weighted percentages"
}
```

---

## Phase 4: Appendix construction

### Task 4.1: Build the Appendix A bug-level table

**Files:**
- Modify: `/Users/nguyenmn/clax/drafts/ai4scientists/main.tex` (Appendix A placeholder from Task 1.4)

- [ ] **Step 1: Verify CHANGELOG line references**

```bash
sed -n '215,300p' /Users/nguyenmn/clax/.claude/worktrees/benchmark-clax-pt/CHANGELOG.md
```

Confirm bugs 1–14 appear in the two PT Bugs tables at lines 218–221 and 260–278, and the architectural-redesign block begins at line 294. If line numbers have shifted (the CHANGELOG is active), update the citation column in Step 3 accordingly.

- [ ] **Step 2: Replace the placeholder with the appendix preamble**

Find `[PLACEHOLDER: populated in Task 4.1.]` and replace with:

```latex
This appendix lists the 15 supervision events documented during the \claxpt v0.1.0 development window ([TBD V0.1.0 DAYCOUNT] active work days, [TBD SESSION COUNT] sessions, ending at the opening of pull request \#9 in the public repository). Each row is one event; the count itself involves a judgment call (a defensible range of 13--15 depending on whether a test-metric correction and the architectural redesign are counted separately---see scope note below). The classification was reviewed by a second pass that independently reconstructed the table from the same development log without seeing the original classification; rows where the two passes agreed are marked confidence~$=$~high, rows where intervention-level attribution required interpretation are marked medium or low. The development log records technical fixes but does not record which party (physicist or agent) proposed each fix; intervention-level attribution therefore rests on the supervisor's recall, cross-checked against the agent's own session-log review.

\paragraph{Scope note.}
The following items are intentionally excluded from the count: (i) test-harness key typos that affected test runs but not the implementation module; (ii) caveats acknowledged in the development log but never escalated to bug status (e.g., the \texttt{rs\_h=99.0} hardcode resolved after v0.1.0; the $\sigma_v^2$ FFTLog-grid integration with $\sim$0.1\% accuracy cost); (iii) post-v0.1.0 fixes (NumPy 2.0 \texttt{np.trapezoid} compatibility; AD-correctness gradient fixes that landed in May 2026). The 12-day window closes at the v0.1.0 milestone; later work on \claxpt extends the public record but is not part of this case study's evidence base.

\begin{table*}[h]
\caption{Issue-level classification for \claxpt v0.1.0 development. Categories: convention/unit (CV), algorithm transcription (AT), numerical coefficient (NC), architectural mismatch (AM), calibration patch (CP), test methodology (TM). Intervention levels: autonomous (A), human-accelerated (HA, physicist supplied a magnitude/shape observation), reconsider-prompt-failed-then-domain-hint (RP--DH), physicist-rejected-agent-solution (PR). Confidence is high (H) where development-log evidence directly supports the assignment, medium (M) where category is clear but intervention required interpretation, low (L) where both passes disagreed or evidence is indirect.}
\label{tab:bug-table}
\centering
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}rlllccl@{}}
\toprule
\# & Issue & Discovered & Category & Intervention & Confidence & Log ref. \\
\midrule
1  & FFTLog $M_{22}$ Hermitian vs symmetric packing & 2026-04-02 & AT & A      & M & L218 \\
2  & $M_{22}$ row-major vs LAPACK column-major      & 2026-04-02 & CV & A      & M & L219 \\
3  & IR resummation log $k$-grid                    & 2026-04-03 & AT & A      & M & L220 \\
4  & IR resummation linear interpolation            & 2026-04-03 & AT & A      & M & L221 \\
5  & Spurious $h^3$ multiply in bias/multipole fns  & 2026-04-04 & CV & A      & M & L274 \\
6  & $b_4$ $k$-factor $(k_h/h)^2 \to k_h^2$        & 2026-04-04 & CV & HA     & M & L275 \\
7  & Incomplete $M_{22}$ RSD kernels                & 2026-04-04 & AT & A      & M & L276 \\
8  & Incomplete $M_{13}$ RSD kernels                & 2026-04-04 & AT & A      & M & L277 \\
9  & UV counterterm coefficients ($\ell=2,4$)       & 2026-04-04 & NC & HA     & M & L278 \\
10 & $P_{gg,\ell=2}$ tree used isotropic component  & 2026-04-09 & AT & A      & M & L264 \\
11 & $P_{gg,\ell=4}$ tree had spurious $b_1$ factors & 2026-04-09 & AT & A     & M & L265 \\
12 & Hexadecapole zero-crossing test metric         & 2026-04-09 & TM & A      & M & L266 \\
13 & $\alpha=0.27$ calibration patch (introduction) & 2026-04-09 & CP & A      & H & L267 \\
14 & $\alpha$ rejection and anisotropic re-derivation & 2026-04-09 & AM & PR  & H & L268 \\
15 & RSD architectural redesign (GL $P(k,\mu)$ assembly) & 2026-04-08 & AM & RP--DH & M & L294--L336 \\
\bottomrule
\end{tabular}
\end{table*}

The two classification passes agreed on category for all 15 rows and on intervention level for 12 rows; the three rows with intervention-level disagreement (\#6 between A and HA; \#9 between A and HA; \#15 between RP--DH and DH-only) are marked confidence M to flag this. Bug \#12 is classified as a test-methodology change rather than an implementation defect; readers preferring a stricter count may exclude it, yielding 14 rows. Bugs \#13 and \#14 represent the same parameter ($\alpha$) added and then removed within a single session 50 minutes apart; we count them as separate events because the fix was a structural re-derivation, not a numerical adjustment. Reasonable taxonomies could collapse these into one row, yielding 13 rows. The text of \cref{sec:bugs} and the abstract report the upper bound (15); readers preferring a stricter count of 13 or 14 should reach equivalent conclusions about the qualitative pattern.

Across the 15 events the intervention-level distribution is: 10 autonomous (A); 2 human-accelerated (HA); 1 architectural with explicit reconsider-prompt-failed-then-domain-hint (\#15); 1 physicist-rejected-agent-solution (\#14); 1 test-methodology correction (\#12). Session-time was distributed unevenly: rows 1--12 collectively consumed substantially fewer sessions than rows 13--15 alone, which together spanned the architectural-wall window described in \cref{sec:wall}.
```

- [ ] **Step 3: Compile and verify**

```bash
cd /Users/nguyenmn/clax/drafts/ai4scientists
pdflatex -interaction=nonstopmode main.tex 2>&1 | tail -10
pdflatex -interaction=nonstopmode main.tex 2>&1 | tail -5
pdflatex -interaction=nonstopmode main.tex 2>&1 | tail -5
pdfinfo main.pdf | grep Pages
```

Expected: Appendix A renders with `table*` (two-column-spanning). Table fits on one page. Total PDF grows by ~1 page (appendix page) — this is outside the 8-page body cap.

- [ ] **Step 4: Commit**

```bash
git -C /Users/nguyenmn/clax add drafts/ai4scientists/main.tex drafts/ai4scientists/main.pdf
git -C /Users/nguyenmn/clax commit -m "paper: Appendix A bug-level classification table"
```

---

## Phase 5: Polish and final

### Task 5.1: Resolve all `[TBD]` markers from Phase 0 data collection

**Files:**
- Modify: `/Users/nguyenmn/clax/drafts/ai4scientists/main.tex`

- [ ] **Step 1: Enumerate remaining TBDs**

```bash
grep -n "TBD" /Users/nguyenmn/clax/drafts/ai4scientists/main.tex
```

Expected: 5–7 matches across abstract, §4.3, Appendix A preamble.

- [ ] **Step 2: Replace each TBD with the resolved value from Phase 0**

Use the "Resolved TBDs" block in this plan (populated by Tasks 0.2 and 0.3). Replace:

| Marker | Replace with |
|---|---|
| `[TBD V0.1.0 DAYCOUNT]` | Integer from Task 0.2 (default `12`) |
| `[TBD SESSION COUNT]` | Integer from Task 0.3 (default `57`) |
| `[TBD STUCK COUNT]` | Integer from Task 0.3 (default `33`) |
| `[TBD PR#9 DATE]` | Not used in body text; appears only in resolved-values block; skip if not in `main.tex` |

If a value is unresolvable (data-collection failed), use the default from the header table and append a footnote: `\footnote{Original-submission figure retained; recomputation from session JSONLs was not feasible during the revision window.}`

- [ ] **Step 3: Add cost sentence to §4.3 Counterexamples paragraph**

Find the `[TBD: insert one sentence on inference cost here per Task 5.1.]` placeholder. Replace with one of:

- If `[TBD TOTAL COST USD]` is resolved to a number `$X.XX`:
  ```
  The total inference cost over the v0.1.0 window was approximately \$X.XX, predominantly Opus 4.x with prompt caching.
  ```

- If unresolved:
  ```
  Inference cost over the v0.1.0 window was not aggregated at the time and is recoverable only approximately from session logs; we omit a precise figure rather than report an unverified one.
  ```

- [ ] **Step 4: Verify zero remaining TBDs**

```bash
cd /Users/nguyenmn/clax/drafts/ai4scientists
pdflatex -interaction=nonstopmode main.tex 2>&1 | tail -5
pdflatex -interaction=nonstopmode main.tex 2>&1 | tail -5
grep -c "TBD" /Users/nguyenmn/clax/drafts/ai4scientists/main.tex
```

Expected: zero.

- [ ] **Step 5: Commit**

```bash
git -C /Users/nguyenmn/clax add drafts/ai4scientists/main.tex drafts/ai4scientists/main.pdf
git -C /Users/nguyenmn/clax commit -m "paper: resolve all TBDs from Phase 0 data collection"
```

### Task 5.2: Add data-availability section + first-mention footnote

**Files:**
- Modify: `/Users/nguyenmn/clax/drafts/ai4scientists/main.tex`

- [ ] **Step 1: Add data-availability section before Acknowledgments**

After the Conclusion (`\section{Conclusion}` ends) and before `\section*{Acknowledgments}`, insert:

```latex
\section*{Code and Data Availability}
\label{sec:availability}

The \claxpt source code, the full \texttt{CHANGELOG.md} development log, the per-session intervention classification underlying Appendix~\ref{app:bug-table}, and the commit history covering the v0.1.0 development window are publicly available at \url{https://github.com/MinhMPA/clax-pt}. The benchmark validation data used in \cref{tab:accuracy} (\classpt reference power spectra at the Planck 2018 fiducial cosmology and at the cosmology variations referenced in \cref{sec:protocol}) is released alongside the code.

```

- [ ] **Step 2: Add a first-mention footnote in §2.1**

In §2.1 "What Was Built," locate the first occurrence of `\claxpt` in the body prose. Change the sentence by appending a footnote at that first mention:

Find:

```latex
One-loop perturbation theory (the next-to-leading-order correction beyond the linear approximation) extends the linear matter power spectrum to mildly nonlinear scales by computing correction terms from second- and third-order density fields, producing predictions for galaxy clustering that are essential for extracting cosmological parameters from spectroscopic surveys~\citep{damico2020, perko2016}. \claxpt implements this calculation in \jax:
```

Replace `\claxpt implements this calculation in \jax:` with:

```latex
\claxpt\footnote{Source code, development log, and supervision data: \url{https://github.com/MinhMPA/clax-pt}; see \cref{sec:availability}.} implements this calculation in \jax:
```

- [ ] **Step 3: Compile and verify**

```bash
cd /Users/nguyenmn/clax/drafts/ai4scientists
pdflatex -interaction=nonstopmode main.tex 2>&1 | tail -5
pdflatex -interaction=nonstopmode main.tex 2>&1 | tail -5
grep -c "MinhMPA/clax-pt" /Users/nguyenmn/clax/drafts/ai4scientists/main.tex
```

Expected: 2 matches (footnote + availability section).

- [ ] **Step 4: Commit**

```bash
git -C /Users/nguyenmn/clax add drafts/ai4scientists/main.tex drafts/ai4scientists/main.pdf
git -C /Users/nguyenmn/clax commit -m "paper: data-availability section and first-mention footnote"
```

### Task 5.3: Update Figure 2 caption

**Files:**
- Modify: `/Users/nguyenmn/clax/drafts/ai4scientists/main.tex` (Figure 2 caption block)

- [ ] **Step 1: Locate the caption**

Find:

```latex
\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{figures/fig2_bug_taxonomy.pdf}
\caption{Bug taxonomy for the \claxpt development. Ten issues were resolved autonomously by the agent iterating against oracle tests. Two were accelerated by the physicist's domain knowledge (unit bugs invisible to shape-based comparisons). Three required essential human physics judgment: an architectural redesign, a fudge-factor rejection, and identification of the correct physical formula.}
\label{fig:taxonomy}
\end{figure}
```

- [ ] **Step 2: Replace the caption**

```latex
\caption{Issue taxonomy for the \claxpt v0.1.0 development. Of 15 documented supervision events, 10 were resolved autonomously by the agent iterating against oracle tests; 2 were accelerated by the physicist's domain knowledge (unit-magnitude and dimensional discrepancies invisible to shape-based comparisons); 3 required essential human judgment (an architectural redesign, a calibration-patch rejection, and identification of the correct anisotropic damping formula). Bug-level classification with confidence flags and independent cross-check provenance is provided in Appendix~\ref{app:bug-table}.}
```

- [ ] **Step 3: Inspect the figure source for visualization coherence**

```bash
file /Users/nguyenmn/clax/drafts/ai4scientists/figures/fig2_bug_taxonomy.pdf
```

The figure is a PDF; we cannot edit visual labels from `main.tex`. If the figure shows session-weighted 80/10/10 percentages (rather than issue-weighted 10/2/3 of 15), flag this for the operator to regenerate before submission. The figure source was presumably a separate plotting script; locating and updating it is outside this task's scope but should be done before camera-ready upload.

- [ ] **Step 4: Compile and commit**

```bash
cd /Users/nguyenmn/clax/drafts/ai4scientists
pdflatex -interaction=nonstopmode main.tex 2>&1 | tail -5
pdflatex -interaction=nonstopmode main.tex 2>&1 | tail -5
git -C /Users/nguyenmn/clax add drafts/ai4scientists/main.tex drafts/ai4scientists/main.pdf
git -C /Users/nguyenmn/clax commit -m "paper: update Figure 2 caption for K2+K4 coherence"
```

### Task 5.4: Generalization-language audit

**Files:**
- Modify: `/Users/nguyenmn/clax/drafts/ai4scientists/main.tex` (abstract, §1, §4, §6)

- [ ] **Step 1: Grep for overgeneralizing phrasing**

```bash
grep -n -E "current LLMs|all agents|in general|broadly diagnostic|the primary factor|the irreducible|principles for the community|address the gap|scaling alone|capabilities not yet exhibited" /Users/nguyenmn/clax/drafts/ai4scientists/main.tex
```

- [ ] **Step 2: For each match, scope-bind or cut**

Transformation rules:
- "current LLMs cannot do X" → "the agent in this case study did not do X" or cut
- "the primary factor in whether the agent's output was trustworthy" → "the primary factor in this case study"
- "principles for supervising AI agents" → "practices that emerged in this project"
- "Closing this gap would require..." → "Closing the gap we observed would require..."
- "not yet exhibited by current LLMs and not obviously addressed by scaling alone" → cut entirely (this is an N=1 claim about LLM capability)

For each grep match, apply the appropriate transformation. When sentences are deleted entirely, ensure the surrounding paragraph still reads cleanly.

- [ ] **Step 3: Verify**

```bash
cd /Users/nguyenmn/clax/drafts/ai4scientists
pdflatex -interaction=nonstopmode main.tex 2>&1 | tail -5
pdflatex -interaction=nonstopmode main.tex 2>&1 | tail -5
grep -c -E "current LLMs|the primary factor in whether|principles for the community" /Users/nguyenmn/clax/drafts/ai4scientists/main.tex
```

Expected: zero matches in the final grep.

- [ ] **Step 4: Commit**

```bash
git -C /Users/nguyenmn/clax add drafts/ai4scientists/main.tex drafts/ai4scientists/main.pdf
git -C /Users/nguyenmn/clax commit -m "paper: scope-bound generalization language (F2)"
```

### Task 5.5: Voice consistency sweep

**Files:**
- Modify: `/Users/nguyenmn/clax/drafts/ai4scientists/main.tex`

- [ ] **Step 1: Audit "we / our"**

```bash
grep -c -E "\bwe\b|\bour\b|\bus\b" /Users/nguyenmn/clax/drafts/ai4scientists/main.tex
```

- [ ] **Step 2: Apply the audit rule**

For each "we / our / us" occurrence:
- **Keep** if it's the conventional scientific plural ("we present X," "we observed Y," "we report Z" — all standard single-author scientific prose).
- **Change to a third-person form** ("the supervisor", "the supervising physicist") ONLY where the sentence claims a specific decision that would mislead a reader into thinking multiple authors made it ("we documented 15 issues and classified each by..." — the classifier was specifically the supervisor; rephrase as "the supervisor documented 15 issues and classified each by...").
- **First-person singular ("I")** is acceptable but should be applied consistently or not at all. Default: keep "we" throughout (scientific convention) and only switch where misattribution is acute.

Most "we" occurrences should remain "we." Pick consistency over volume.

- [ ] **Step 3: Compile and verify**

```bash
cd /Users/nguyenmn/clax/drafts/ai4scientists
pdflatex -interaction=nonstopmode main.tex 2>&1 | tail -5
pdflatex -interaction=nonstopmode main.tex 2>&1 | tail -5
```

- [ ] **Step 4: Commit (only if non-trivial changes were made)**

```bash
git -C /Users/nguyenmn/clax diff --quiet drafts/ai4scientists/main.tex || {
  git -C /Users/nguyenmn/clax add drafts/ai4scientists/main.tex drafts/ai4scientists/main.pdf
  git -C /Users/nguyenmn/clax commit -m "paper: voice consistency sweep"
}
```

### Task 5.6: Final compile, page-count check, visual diff, and tag

**Files:**
- Read: `/Users/nguyenmn/clax/drafts/ai4scientists/main.pdf` vs `main.pdf.before-revision`

- [ ] **Step 1: Full compile chain**

```bash
cd /Users/nguyenmn/clax/drafts/ai4scientists
pdflatex -interaction=nonstopmode main.tex 2>&1 | tail -10
bibtex main 2>&1 | tail -5
pdflatex -interaction=nonstopmode main.tex 2>&1 | tail -5
pdflatex -interaction=nonstopmode main.tex 2>&1 | tail -5
```

Expected: no fatal errors. `bibtex` may emit warnings about unused entries — non-fatal.

- [ ] **Step 2: Check page distribution**

```bash
pdfinfo main.pdf | grep Pages
pdftotext -layout main.pdf - | awk 'BEGIN{p=1} /\f/{p++} /^[ ]*References|^[ ]*Acknowledgments|^[ ]*A\.|^[ ]*Appendix|^[ ]*[0-9]\.\s/{print "page "p": "$0}'
```

Expected: body content (numbered sections §1–§6) ends on or before page 8. Acknowledgments + Appendix A + References fall on pages 9+. The 8-page body cap is respected (this is the venue's hard constraint).

If body exceeds 8 pages, return to Phase 5 and apply additional cuts: trim §1 Introduction redundancy with abstract, trim §4.2 Responsibility paragraph further, trim §4.3 Two remediation paths paragraph.

- [ ] **Step 3: Visual diff against pre-revision baseline**

```bash
pdftotext -layout main.pdf - | head -50
echo "---PRE-REVISION---"
pdftotext -layout main.pdf.before-revision - | head -50
```

Spot-checks: title shows question mark + "A Case Study"; author block shows only Nhat-Minh Nguyen with three affiliations; abstract reads the revised version; §4 shows three subsections with new headings; appendix table is present.

- [ ] **Step 4: Final commit and cleanup**

```bash
git -C /Users/nguyenmn/clax rm drafts/ai4scientists/main.pdf.before-revision
git -C /Users/nguyenmn/clax add drafts/ai4scientists/main.tex drafts/ai4scientists/main.pdf
git -C /Users/nguyenmn/clax commit -m "paper: final compile (revised camera-ready)"
```

- [ ] **Step 5: Tag the revision**

```bash
git -C /Users/nguyenmn/clax tag -a paper-ai4sci-2026-revised -m "AI4Science 2026 paper, revised camera-ready (post-reviews)"
```

---

## Self-Review Checklist (run after writing this plan; fix any issues inline)

**1. Spec coverage** — every grilling-session decision maps to a task above:

- [x] Title change → Task 1.1
- [x] Author block (single-author, ORCID, ICML syntax) → Task 1.1
- [x] Acknowledgments → Task 1.2
- [x] §4 restructure (S2) → Task 1.3
- [x] Appendix scaffold → Task 1.4
- [x] §3.2 L1/L2/L3 rewrite + factual correction → Task 2.1
- [x] Positive-autonomy paragraph → Task 2.2
- [x] §3.3 50-minute window detail → Task 2.3
- [x] Multi-cosmo counterfactual rewrite → Task 2.4
- [x] §4.2 trim "Governance" → "Responsibility" → Task 2.5
- [x] §4.3 "What This Case Study Cannot Show" → Task 2.6
- [x] Cut Boltzmann/clax typo passage → Task 2.8
- [x] Conclusion rewrite → Task 2.9
- [x] Abstract rewrite → Task 3.1
- [x] 80/10/10 scrub → Task 3.2
- [x] Appendix A bug table → Task 4.1
- [x] TBD resolution → Task 5.1
- [x] GitHub URL + data-availability → Task 5.2
- [x] Figure 2 caption update → Task 5.3
- [x] Generalization-language audit → Task 5.4
- [x] Voice sweep → Task 5.5
- [x] Final compile + page check + tag → Task 5.6

**2. Placeholder scan** — `[TBD ...]` markers are explicit and have explicit resolution instructions in the header table and Task 5.1. No bare "TODO" / "implement later" / "fill in details" exists.

**3. Type consistency** — section labels (`sec:lessons`, `sec:managing`, `sec:credit`, `sec:cannot`, `sec:availability`, `app:bug-table`, `tab:bug-table`) are used consistently across Tasks 1.3, 2.5, 2.6, 5.2. The L1/L2/L3 vocabulary appears once in §4.3 (Task 2.6) and as substance (without the L-labels) in §3.2 (Task 2.1) — this is intentional: the labels make sense in the limitations section where the framework is named explicitly, but cluttering §3.2 with them would be over-jargon.

**Data-collection items** (Tasks 0.2, 0.3) have fallback defaults in the header table; the plan can complete even if data collection fails, with a small precision loss documented in footnotes.

---

## Execution Notes

- Phase 0 Tasks 0.2 and 0.3 can run in parallel with Phase 1 — start them immediately and circle back at Task 5.1.
- Phase 1 must complete before Phase 2 (structure first, then content).
- Phase 2 Tasks 2.1–2.9 are mostly independent; they can run in parallel if dispatched to multiple subagents.
- Phase 3 and Phase 4 can run in parallel after Phase 2.
- Phase 5 runs last.

Each task ends with a commit. If a task fails its compile, do not amend the previous commit — create a new commit fixing the issue.

---

## Resolved TBDs (filled during execution)

**From Task 0.2 (run 2026-05-20):**
- PR#9 (on `smsharma/clax`, branch `feat/clax-pt`) `createdAt`: **2026-05-01T10:28:53Z**
- v0.1.0 start (first `clax/ept.py` commit): **2026-03-29 00:14:57 +0900**
- v0.1.0 functional milestone (`PR prep` commit): **2026-04-12 16:32:19 +0900**
- Active work days (Mar 29 – Apr 12, business days): **12** (matches original submission)
- Note: PR#9 opened 19 days after the v0.1.0 milestone, during paper-prep / pre-release polish; the 12-day count bounds the development sprint, not the time until public release.

**From Task 0.3 (deferred to dev machine, run 2026-05-21):**
- Session count (verified): *unresolved on this machine; fallback `57` from original submission*
- Stuck-session count (verified): *unresolved; fallback `33` from original submission*
- Total inference cost (Opus + Sonnet): *unresolved; §4.3 cost sentence uses the "not aggregated" fallback*
- α=0 probe initiation: *unresolved; defaults to physicist-initiated per grilling Q-A, matches original submission*
- L1 phrasing attempts: *unresolved; §3.2 prose says "the physicist explicitly attempted process scaffolding" without quantifying*

User will rerun Task 0.3 on the dev machine and either confirm the fallbacks or open a follow-up commit updating the manuscript with verified numbers.
