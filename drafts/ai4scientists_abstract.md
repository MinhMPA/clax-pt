# AI Scientists Workshop Submission — Abstract Draft

**Workshop**: AI Scientists – Tools, Co-authors, or Founders? (ICML 2026)
**Track**: Position

---

## Title

Oracle Tests Are Not Enough: Lessons from Building Scientific Software with an AI Agent

## TL;DR

An AI agent resolved 11 of 14 bugs autonomously while building a validated one-loop perturbation theory code for galaxy clustering, but the three it couldn't — a fundamentally wrong architecture, a fudge factor passing all tests, and the diagnosis that exposed it — were invisible to oracle testing and required human physics judgment. Supervision protocol design mattered more than model capability.

## Abstract

Are AI coding agents tools, co-authors, or autonomous scientists? We present a quantified case study: a physicist supervising an AI agent (Claude Code, Opus 4.6) over 12 days and 57 sessions to build a differentiable one-loop perturbation theory module for galaxy clustering in JAX (~2,100 lines, validated to sub-percent accuracy against CLASS-PT). We documented 14 bugs and classified each by whether autonomous resolution was possible.

The agent resolved 11 autonomously — convention errors, algorithm transcription, numerical coefficients — by iterating against oracle test suites. The three it could not all eluded oracle detection, and shared a common property: the agent treated symptom reduction as equivalent to root-cause resolution. It spent 33 of the 57 sessions tuning coefficients within a fundamentally wrong code architecture, unable to recognize the problem was structural rather than parametric. After the physicist directed a redesign, the agent then found a scalar correction by grid search that reduced error below 1% across all test cases — but the value corresponds to no physical quantity and would silently produce wrong predictions at any other cosmology. The physicist diagnosed the fudge factor by asking a question the agent could not formulate: "does this parameter correspond to anything in the reference code?"

Three supervision practices, developed iteratively, we found critical for catching what oracle tests missed: testing against diverse cosmologies beyond the fiducial calibration; shared changelogs that surfaced stalled exploration across sessions; and an explicit rule against unphysical numerical patches. In our case study, the design of these supervision protocols — not model capability — determined whether the agent's output was trustworthy. Closing this gap would require agents that can propose architectural alternatives rather than optimizing within a given structure, and distinguish predictive adequacy from explanatory correctness — capabilities not addressed by scaling alone.
