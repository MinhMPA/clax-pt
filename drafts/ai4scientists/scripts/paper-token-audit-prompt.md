# Agent prompt — clax-pt paper token audit

Paste the block below into a Claude Code session on the machine that actually ran the clax-pt bring-up (the Mar 29 → Apr 12 work). The Mac that drafted the paper does **not** have those transcripts; only the bring-up machine does.

---

## Prompt

```
Task: produce auditable token-usage and cost numbers for the clax-pt paper appendix.

The paper at drafts/ai4scientists/main.tex claims "12 days and 57 sessions" for
the clax-pt bring-up and "33 of 57 sessions" requiring physicist supervision.
Reviewer 1 objected that the supervision split is "post-hoc rationalization."
Your job is to substantiate (or refute) those numbers from the local Claude
Code session transcripts on THIS machine.

Setup:
  1. Pull the paper_drafts branch:
       cd ~/clax            # or wherever the clax repo lives on this machine
       git fetch origin
       git checkout paper_drafts
       git pull origin paper_drafts
  2. The audit script is at drafts/ai4scientists/scripts/count_paper_tokens.py.

Run two windows so we can report sensitivity:
  A. Default window — first clax-pt commit through PR prep:
       python3 drafts/ai4scientists/scripts/count_paper_tokens.py \
         --start 2026-03-29 --end 2026-04-12 \
         --output-json paper_tokens_A.json \
         --output-csv  paper_tokens_per_session_A.csv

  B. Intensive-phase window — bug-fixing only:
       python3 drafts/ai4scientists/scripts/count_paper_tokens.py \
         --start 2026-04-02 --end 2026-04-12 \
         --output-json paper_tokens_B.json \
         --output-csv  paper_tokens_per_session_B.csv

What to report back (target: ~300 words + a 2-row table):

  1. SESSION COUNT
     - File count in each window vs paper's claim of 57.
     - If off by >5 sessions, say so and propose what the discrepancy means
       (retention gap, sessions on a different machine, sessions aborted before
       producing usage data, etc.).

  2. TOTAL COST in USD
     - For both windows.
     - Broken down by Opus 4.7 / Opus 4.6 / Sonnet / Haiku if multiple present.
     - Flag any model with `unknown` pricing in the output (script will warn).

  3. INTERVENTION HEURISTIC
     - The script flags sessions where the USER message text matched a regex
       of physics-domain keywords (anisotropic, sigmatot, fudge, multipole,
       etc.). This is a LOWER BOUND on supervision-heavy sessions.
     - Report: heuristic count vs paper's "33 of 57". Don't try to make them
       match — if the heuristic gives a different number, that itself is the
       data point Reviewer 1 wants.

  4. CAVEATS to document
     - Was JSONL retention truncated? (Compare earliest file timestamp to the
       2026-03-29 paper claim.)
     - Were sessions on the no-charge model tier? (Look for model="<synthetic>"
       or missing usage records in the script's no_usage_data count.)
     - Anthropic pricing during the actual window — VERIFY the script's
       PRICING table is correct for Mar 29 → Apr 12. Note any discrepancy
       and rerun if needed.
     - Multiple machines? If you ran sessions on multiple hosts (laptop +
       cluster + remote dev box), this audit only sees ONE machine's data.
       Report which machine this is and whether you suspect others contributed.

Hard rules:
  - DO NOT modify the script's PRICING constants without first checking
    Anthropic's published pricing for the window. If pricing changed
    mid-window, split the calculation and document the source.
  - DO NOT exclude sessions to make the count match 57. Report the raw count.
  - DO NOT use an LLM to classify sessions for the "33 of 57" claim. The
    keyword heuristic is the only automated estimate; honest answer is "the
    heuristic says X; manual classification needed for the paper's 33 claim."
  - Commit the output JSON+CSV files to paper_drafts when done:
       git add paper_tokens_*.json paper_tokens_per_session_*.csv
       git commit -m "audit: token usage for clax-pt paper appendix"
       git push origin paper_drafts

Final deliverable: one paragraph suitable for pasting into the paper's
methods/appendix, citing the exact numbers and flagging the heuristic vs
manual-classification distinction explicitly.
```

---

## What you'll get back

Two artifacts in your `clax/` repo on the remote machine:

- `paper_tokens_A.json`, `paper_tokens_per_session_A.csv` — the full window
- `paper_tokens_B.json`, `paper_tokens_per_session_B.csv` — the intensive phase

Plus an agent narrative that:

- Confirms or flags the "57 sessions" claim.
- Gives a dollar figure suitable for the paper appendix.
- Provides a heuristic intervention count, with explicit caveat that the paper's "33" needs manual validation, not automation.
- Lists any environmental caveats (multi-machine, retention, pricing).

## After the audit returns

The two windows let you write something like:

> "Across the 11 active development days (2026-04-02 to 2026-04-12), the
> clax-pt bring-up consumed $X across N Claude Code sessions and M
> conversational turns. A keyword-based heuristic flagged K of N sessions as
> involving physics-domain user input; this is a lower bound on the
> supervision rate, and the manual classification reported in §4 (33 of 57)
> remains the authoritative figure."

The numerator/denominator difference between heuristic-K-of-N and paper-33-of-57 is itself the inter-rater check Reviewer 1 asked for.

## Why not on the Mac that drafted the paper

The Mac that hosts `main.tex` was active 2026-04-15 → present (post-bring-up, paper-writing and cleanup phase). Token-insights on that machine give $7.9k clax/clax-pt cost — but that's a *different* window than the paper's claim, so those numbers can't substantiate the "57 sessions" figure. Only the bring-up machine has those JSONLs.
