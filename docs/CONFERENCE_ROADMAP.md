# Conference Readiness Roadmap

**Status:** Research planning; no results or submission claims.  
**Companion:** [Initial paper draft](PAPER_DRAFT.md) | [Project plan](PROJECT_PLAN.md) | [Issues](https://github.com/jcallinan/panther-lake-ai-lab/issues)

## Target contribution and positioning

**Primary thesis:** Practical heterogeneous inference on consumer AI PCs is governed by runtime compatibility, verified execution placement, workload characteristics, and energy constraints—not peak advertised TOPS alone.

**Candidate submission types:** reproducibility/artifact track, applied AI/edge systems workshop, performance evaluation workshop, or full empirical systems conference paper after sufficient evidence. Select venues based on current calls for papers, peer-review requirements, paper length, reproducibility expectations, and deadlines; do not assume eligibility or acceptance.

## Work packages and exit gates

| Stage | Deliverable | Exit gate |
|---|---|---|
| A. Literature and novelty | 20–30 verified peer-reviewed references, closest-work matrix | Clear research gap and falsifiable RQs |
| B. Protocol | Preregistered models, datasets, precision, metrics, power method, trial plan | Protocol reviewed before measurements |
| C. Infrastructure | Version-pinned scripts, backend verification, result schema | Re-run from clean environment |
| D. Pilot | One vision model, one embedding model, one small LLM | Correctness and variance characterized |
| E. Main study | CPU/GPU/NPU where compatible; battery and sustained tests | Complete raw results and failure records |
| F. Validation | Repeated trials, CIs, independent reproduction | Figures regenerate from raw data |
| G. Paper | Full introduction, related work, methods, results, discussion, threats | Internal review and citation audit |
| H. Submission | Venue-specific manuscript, anonymized artifact if required | Checklist complete; submitted by human author |

## Minimum publishable experiment

1. Freeze OS, firmware, runtime and driver versions.
2. Validate CPU/GPU/NPU enumeration and actual execution.
3. Run **at least two representative model families** across all *compatible* devices, recording unsupported combinations.
4. Record correctness, latency distributions, throughput, memory, and defensible power telemetry.
5. Repeat across multiple sessions; report confidence intervals and negative results.
6. Publish all code, configurations and sanitized raw results.
7. Seek replication on a second device or additional Panther Lake SKU to strengthen external validity.

## Risks and mitigations

- **NPU model compilation fails:** treat as a documented compatibility finding, not an omitted trial.
- **Runtime silently falls back:** verify device assignment and inspect execution traces/logs.
- **Power data unreliable:** use calibrated measurement or exclude strong energy claims.
- **Single laptop:** explicitly limit generalizability and invite cross-device reproduction.
- **Software changes rapidly:** pin versions and compare version blocks separately.
- **Unclear novelty:** prioritize operator coverage, fallback diagnostics, and energy/latency Pareto analysis over generic benchmark rankings.
- **Model licensing/privacy:** use public permissible datasets, hashes and redacted logs.

## Proposed 16-week research schedule

- **Weeks 1–2:** literature review, novelty map, research questions, venue shortlist.
- **Weeks 3–4:** experimental protocol, scripts, hardware baseline and validation.
- **Weeks 5–6:** pilot models, correctness, telemetry and statistical power planning.
- **Weeks 7–10:** main measurements and compatibility/failure taxonomy.
- **Weeks 11–12:** battery, thermal, memory and cross-session replication.
- **Weeks 13–14:** analysis, figures, limitations and manuscript results.
- **Weeks 15–16:** external review, reproducibility audit, venue formatting and submission preparation.

This is a planning cadence, not a promise of conference acceptance.

## Research integrity checklist

- [ ] No fabricated benchmark results or citations
- [ ] All paper claims trace to raw measurements or cited sources
- [ ] Distinguish device detection, compilation, execution and sustained benchmark success
- [ ] Publish failures and fallback behavior
- [ ] State limitations of single-device experiments
- [ ] Disclose relevant funding, conflicts and AI-assisted writing according to venue policy
- [ ] Obtain necessary approvals for any human-subject or sensitive-data work (not anticipated for standard hardware benchmarking)
