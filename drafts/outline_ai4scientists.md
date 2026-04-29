# Paper 2: AI Scientists Workshop (ICML 2026) — Position Track

## Title

**The Scientist Strikes Back: Where Human Physics Judgment Saved (and Failed) an AI Agent Building a Cosmological Perturbation Theory Code**

## Framing

Position/experience paper. The contribution is the empirical record of what a physicist-AI collaboration actually looked like on a real scientific software project, and the lessons for the "AI scientist" question. Not a methods paper — the code is the laboratory, not the deliverable.

---

## Abstract sketch (150 words)

We report on the development of clax-pt, a differentiable one-loop perturbation theory module for galaxy clustering, built over 12 days through 57 agent sessions with Claude Code (Opus 4.6). The project provides an empirical case study for the "AI as scientist" question: the agent autonomously produced ~2,100 lines of working scientific code, yet required 14 bug-fixing interventions — some of which it could have found independently and some that were fundamentally inaccessible to it. We categorize these interventions along an autonomy spectrum: (1) bugs the agent would have found given more time (matrix packing, kernel coefficients); (2) bugs requiring domain judgment the agent lacked (unit conventions, fudge-factor temptation); (3) architectural decisions only a physicist could make (when to abandon an approach, what constitutes a valid accuracy metric). We argue that current AI agents are best understood as domain-supervised research assistants, not autonomous scientists, and that the quality of the supervision protocol matters more than the quality of the model.

---

## Section outline

### 1. Introduction (1 page)

- The "AI scientist" narrative ranges from tool to co-author to autonomous discoverer. Empirical evidence is scarce — most claims are based on benchmarks (SWE-Bench, ML-Bench), not sustained scientific projects.
- We offer a detailed case study: building a real scientific code (one-loop perturbation theory for galaxy clustering) with an AI coding agent, over 12 days, producing code that passes sub-percent accuracy validation against an established reference (CLASS-PT).
- Our contribution is not the code itself (described elsewhere) but the **supervision record**: what the human did, what the agent did, and where the boundary was.
- Central claim: the supervision protocol (methodology, test discipline, oracle-driven validation) was the primary determinant of code quality — not the model capability.

### 2. The project and its methodology (1 page)

#### 2.1 What was built
- clax-pt: 1-loop EFT power spectra (P_mm, P_gg, P_gm, RSD multipoles) in JAX.
- 2,100 lines, 14 bugs found and fixed, 9 output spectra validated to sub-percent.
- Brief technical summary (1 paragraph) — point to the companion paper for details.

#### 2.2 The supervision protocol
- Adapted from Carlini (2026) — Anthropic's C-compiler agent project lessons.
- Key principles encoded in CLAUDE.md before any code was written:
  - CLASS-PT as oracle (tests before implementation)
  - CHANGELOG as shared memory across sessions
  - "No fudge factors" rule
  - Parallel agent sessions for competing hypotheses
  - Context-window hygiene (concise test output)
- This protocol was the human's main contribution — not individual code edits.

### 3. The autonomy spectrum: a bug taxonomy (3 pages)

#### 3.1 Category A — Bugs the agent would have found (given time)

**Bugs 1-4: FFTLog and IR resummation (Mar 30-Apr 3)**
- M22 Hermitian vs symmetric (wrong conjugate in matrix loading)
- LAPACK column-major packing convention
- DST k-grid: logspace vs linspace
- Odd/even spline interpolation for BAO mode removal

These had clear symptoms (wrong magnitudes, P_mm error > 1%), clear diagnostics (compare intermediate values against CLASS-PT), and clear fixes (match the convention exactly). The agent was methodically narrowing down when the human parallelized the search — the human contribution was speed, not insight.

**Lesson**: For bugs with observable symptoms and a reference oracle, the agent is competent. Human value-add is parallelization and prioritization, not diagnosis.

#### 3.2 Category B — Bugs requiring domain judgment

**Bug 5: Spurious h³ factor in all bias functions (Apr 4)**
- Every output function multiplied by h³ before return. The code ran without errors, produced plausible-looking curves — only the absolute normalization was wrong.
- The agent had been comparing shapes (slopes, BAO features) and passing tests. The human spotted the magnitude discrepancy because they knew what P(k) should look like at k=0.1 h/Mpc.
- The agent could not have found this from shape comparisons alone. It required knowing the expected order of magnitude — domain knowledge not encoded in the test suite.

**Bug 6: Wrong b₄ k-factor — (k_h/h)² vs k_h² (Apr 4)**
- Unit confusion between h/Mpc and 1/Mpc. Same failure mode: the code runs, the shapes look right, only the scaling is wrong.

**Bug 14: The fudge-factor temptation (Apr 9)**
- The tree-level BAO correction used an empirical alpha=0.27 to minimize errors across 9 spectra simultaneously. This was an implicit fudge factor — it worked numerically but didn't correspond to any physical formula.
- The human recognized this as a fudge, diagnosed the root cause (isotropic approximation of an anisotropic integral), and directed the agent to implement the correct GL-quadrature approach.
- **This is the most important intervention in the project.** The agent had a working solution (alpha=0.27, all tests pass). The human rejected it on physics grounds and demanded the correct solution.

**Lesson**: Unit conventions and fudge factors are invisible to test suites that check only relative agreement. The physicist's role is to reject numerically-valid-but-physically-wrong solutions.

#### 3.3 Category C — Architectural decisions

**The RSD redesign (Apr 8-9)**
- After 4 days of incremental fixes, RSD multipole errors were still 2-8%. The human decided to abandon the incremental approach and redesign: assemble P(k,μ) and GL-integrate, matching CLASS-PT's AP path exactly.
- The agent had been applying local fixes (adjust this coefficient, add this term). The human recognized the errors were structural, not parametric.
- After the redesign, all 9 spectra passed in one session.

**When to stop debugging and start fresh**
- The agent has no concept of diminishing returns. It will continue applying local patches indefinitely. The human must decide when a structural change is cheaper than more patches.

**Lesson**: Knowing when to abandon an approach is currently a uniquely human capability. The agent optimizes within an architecture; the human decides between architectures.

### 4. The supervision protocol as the key variable (1 page)

#### 4.1 What the protocol gave us
- **Oracle-driven development**: Every claim tested against CLASS-PT reference data. Without this, the agent's output would be untestable.
- **Shared memory via CHANGELOG**: Prevented the 57 sessions from repeating failed approaches. The agent cannot remember across sessions — the protocol compensates.
- **"No fudge factors" rule**: Prevented the agent from settling on numerically-adequate-but-physically-wrong solutions (Bug 14).
- **Parallel sessions**: Enabled competing hypotheses without the human needing to do the work themselves.

#### 4.2 What would have happened without it
- Without the oracle: the code would compile and produce plausible output, but accuracy would be unvalidated. The agent has no independent way to know if P_gg(k) is correct.
- Without CHANGELOG: later sessions would re-explore the DST k-grid bug, the M22 packing bug, etc.
- Without "no fudge factors": Bug 14 (alpha=0.27) would have shipped as the final solution.

#### 4.3 Implications for the "AI scientist" question
- The protocol, not the model, is the bottleneck. A better model with a worse protocol would produce worse science.
- This suggests the right framing is not "tool vs co-author vs founder" but **"what supervision protocol enables reliable scientific output?"**

### 5. Lessons and recommendations (1 page)

1. **AI agents are domain-supervised research assistants.** They execute within a well-defined methodology but cannot set that methodology themselves.
2. **The oracle is everything.** Without a known-good reference, the agent cannot validate its own output. For new science (no oracle), the agent is much less useful.
3. **Fudge-factor detection requires domain expertise.** This is perhaps the hardest capability gap: the agent will happily produce a numerically valid but physically meaningless solution.
4. **Shared memory protocols compensate for session amnesia.** CHANGELOG, CLAUDE.md, and supervision logs are infrastructure, not overhead.
5. **Parallel sessions are the human's superpower.** The agent does the work; the human decides what work to do in parallel.
6. **Know when to redesign, not patch.** The human's ability to recognize structural vs parametric failure is currently irreplaceable.

### 6. Conclusion (0.5 pages)

- The "AI scientist" question is premature. The right question is: what supervision protocols enable AI agents to produce reliable scientific software?
- Our case study suggests that with the right protocol, a physicist-AI team can produce validated scientific code ~10x faster than the physicist alone — but the physicist's judgment remains load-bearing at critical junctures.
- We release the full supervision log, CHANGELOG, and commit history as empirical data for the community.

---

## Figures

1. Timeline: 12-day development arc (commits, bugs found, accuracy achieved) — the accuracy convergence plot from the blog post
2. Bug taxonomy visualization: 14 bugs placed on the autonomy spectrum (A/B/C)
3. The fudge-factor case study: alpha=0.27 vs GL-quadrature accuracy comparison
4. Session graph: 57 sessions, showing parallel branches and merge points

---

## Key selling points for this workshop

- **Human-AI co-authorship theme**: Direct empirical evidence of what "co-authorship" looks like in practice — not a benchmark, but a real 12-day project.
- **Position track fit**: This is a forward-looking argument about supervision protocols, not a technical methods contribution.
- **Concrete and reproducible**: The full commit history, CHANGELOG, and supervision log are public. Reviewers can verify every claim.
- **Actionable lessons**: Not "AI is/isn't ready for science" but "here is the protocol that makes it work, and here are the failure modes to watch for."
