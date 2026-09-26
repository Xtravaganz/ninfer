# Context cache can drop from ~109k prefix reuse to full root re-prefill on the next append-only turn

## Summary

I am investigating a context-cache failure in NInfer during a single long-running, monotonically growing chat conversation.

The expected behavior is straightforward:

```text
Turn N:
prompt ~= 109k tokens
-> ~=109k prefix-cache hit

Turn N+1:
prompt ~= 111k tokens
-> reuse the previous ~=109k prefix
-> prefill only the ~=2k-token suffix
```

Instead, after context reuse has been working correctly for many turns, the next normal append-only turn can abruptly lose the entire reusable prefix:

```text
Turn N:
almost full cache hit
-> private endpoint reuse

Turn N+1:
prefix_cache_hit_tokens = 0
computed_prefill_tokens = full prompt
prefix_reuse_path = root
```

The central reproduction is:

```text
109545 prompt tokens
-> 109169 reused
-> private_endpoint

immediately followed by:

111258 prompt tokens
-> 0 reused
-> 111258 prefilled
-> root
```

This has now reproduced with more than one model, so it does not appear to be model-specific.

The next useful step is not another retention-policy change. The materialization planner / resource manager needs enough diagnostics to answer one specific question:

> Did the ~109k reuse candidate still exist when the failing request was planned, and if so, what exact cost / feasibility / portfolio decision caused it to lose to root?

Without that information, several distinct failure modes currently collapse into the same final observation (`prefix_reuse_path = root`).

---

## Reproduction

### Request #8 — healthy reuse

```json
{
  "request_id": 8,
  "result": {
    "prompt_tokens": 109545,
    "computed_prefill_tokens": 376,
    "prefix_cache_hit_tokens": 109169,
    "prefix_reuse_path": "private_endpoint",
    "finish_reason": "stop_token"
  },
  "materialization": {
    "budget_exhausted": false,
    "stop_reason": "insufficient_expected_gain",
    "search_elapsed_ns": 4998905,
    "search_granted_ns": 5000000,
    "search_renewals": 0,
    "search_discovery_used": false,
    "search_boundary_limited": false,
    "targets_evaluated": 54,
    "selected_degradation_units": 0,
    "selected_maximal_fallback": false,
    "initial_predicted_total_ns": 503030213,
    "predicted_total_ns": 503030213
  }
}
```

### Request #9 — immediate cache collapse

The very next normal user turn:

```json
{
  "request_id": 9,
  "result": {
    "prompt_tokens": 111258,
    "computed_prefill_tokens": 111258,
    "prefix_cache_hit_tokens": 0,
    "prefix_reuse_path": "root",
    "finish_reason": "stop_token"
  },
  "materialization": {
    "budget_exhausted": false,
    "stop_reason": "insufficient_expected_gain",
    "search_elapsed_ns": 4999026,
    "search_granted_ns": 5000000,
    "search_renewals": 0,
    "search_discovery_used": false,
    "search_boundary_limited": false,
    "targets_evaluated": 36,
    "selected_degradation_units": 1,
    "selected_maximal_fallback": false,
    "initial_predicted_total_ns": 833423220301,
    "predicted_total_ns": 79156901670
  }
}
```

So the transition is:

```text
Request #8
prompt: 109545
reuse:  109169
path:   private_endpoint

Request #9
prompt: 111258
reuse:  0
path:   root
```

The incoming prompt is still the same monotonically growing conversation.

---

## Environment

The workload is intentionally simple:

- one user
- one long-running conversation
- same conversation/session routed to the same GPU
- no unrelated multi-user cache pressure
- append-only / monotonically growing chat history

Relevant server configuration:

```text
--max-context 256000
--kv-capacity 256000
--max-concurrency 2
--prefill-chunk 1024
--kv-dtype nvfp4

--device-state-slots 2
--host-state-slots 16
--host-kv-mib 8192

--max-private-continuations 8
--max-shared-prefixes 4

--spec mtp
--draft-tokens 4
--lm-head-draft
--preserve-thinking
```

Typical startup memory state is approximately:

```text
KV capacity: 256000 tokens, nvfp4
runtime memory: ~5.59 GiB
free VRAM: ~1.25 GiB
```

`--device-state-slots 2` is therefore intentional and cannot simply be increased without materially changing the memory layout of the workload.

---

## Current test branch

The reproduction is currently being investigated on:

```text
repository:
Xtravaganz/ninfer

branch:
rolling-context-cache

head:
86b929b938f2d7b0c4d1b7215728eac781887582
```

The branch is based on:

```text
54eb0cd47daec4b42ba094b933773cf16e7f2772
```

The running container used for the rolling-policy validation was:

```text
ninfer-lab/ai-ninfer:rolling-context-cache
```

The running process command line was checked directly via `/proc/1/cmdline`, so the policy described below was confirmed to be active during the live test.

---

## Rolling retention was tested and is not sufficient

A strong earlier hypothesis was that long append-only conversations lose their newest useful frontier because the candidate does not inherit the demonstrated demand of the exact ancestors that it extends.

This is related to upstream issue:

- [#180 — Proposal: opt-in rolling context-cache retention for long agent sessions](https://github.com/Neroued/ninfer/issues/180)

A port of that behavior was implemented on the current NInfer tree behind:

```text
--context-cache-policy rolling
```

with `default` remaining unchanged.

The core behavior is:

```cpp
.candidate_demand_mask =
    committed_demand_mask_for(scenario.assessment.shortlist_key) |
    rolling_inherited_demand(lane),
```

`rolling_inherited_demand()` walks committed demand records from the same reuse domain and inherits committed demand from exactly resident ancestors that the current request proved it extends.

Regression coverage includes:

```text
test_rolling_default_inherits_no_ancestor_demand
test_rolling_continuation_inherits_ancestor_demand
test_rolling_unrelated_lineage_inherits_nothing
test_rolling_inherited_demand_flips_the_portfolio_verdict
```

All tests pass.

The important result is that the original cache-collapse behavior still occurs with a different model while rolling retention is definitely active.

Therefore:

```text
rolling retention alone does not prevent this failure
```

This does not mean rolling retention is incorrect or unrelated to other long-session behaviors. It only means it is not sufficient to explain this particular warm -> root transition.

---

## Why the materialization planner is now the main suspect

Request #9 is particularly interesting:

```text
budget_exhausted: false
stop_reason: insufficient_expected_gain

targets_evaluated: 36
selected_degradation_units: 1

initial_predicted_total_ns:
833423220301

predicted_total_ns:
79156901670
```

Approximately:

```text
initial predicted total: 833 s
selected predicted total: 79 s
```

At the same time, the immediately preceding request had proven that a ~109k-token prefix was reusable.

This leaves two fundamentally different classes of failure.

### Case A — the good ~109k candidate still exists

Example:

```text
candidate:
reuse_path = private_endpoint
matched/reused tokens ~= 109169
remaining prefill ~= 2089
...
rejected / not selected because X

root:
reused tokens = 0
remaining prefill = 111258
...
selected
```

If this is what happened, the bug is likely in one of:

```text
cost prediction
portfolio value
pressure/degradation
search ordering
search budget
target assessment
final selection
```

### Case B — the ~109k candidate is already gone

Example:

```text
root candidate exists

but no candidate corresponding to the previous 109169-token frontier
ever reaches MaterializationPlanner
```

Then the bug is earlier:

```text
prefix-index discovery
shortlist-key matching
catalog residency
session/reuse-domain matching
active-edge filtering
inspect_admission()
checkpoint lifecycle
materialization lifecycle
```

Current request logs do not distinguish these cases.

---

# Relevant NInfer classes and code paths

The current tree already has a clean architectural boundary between logical cache discovery, physical admission assessment, pressure search, portfolio valuation and final sealing.

That makes it possible to instrument this without changing policy.

## 1. `ResourceManager<ModelContract>`

Current path:

```text
src/runtime/engine/context_cache/resource_manager.h
```

Relevant methods / structures:

```cpp
template <class ModelContract>
class ResourceManager
```

### `ResourceManager::inspect()`

This is where the root candidate is created and cache-resident candidates are discovered.

Current high-level flow:

```text
rebuild_prefix_index()
    |
    +-> create root AdmissionCandidate
    |
    +-> iterate prefix_index_
            |
            +-> valid_prefix_index_entry()
            +-> base.prefix_shortlist_key(...)
            +-> catalog/shared-catalog state checks
            +-> active-edge checks
            +-> program.inspect_admission(...)
            +-> append Candidate
    |
    +-> plan_materialization(...)
```

Important current filtering points include:

```cpp
if (!valid_prefix_index_entry(index)) {
    continue;
}

if (!incoming || *incoming != index.key) {
    continue;
}
```

Private candidate filtering also includes:

```cpp
if (entry.state != CatalogState::Catalogued ||
    !entry.handle ||
    private_has_active_edge(index.slot)) {
    continue;
}
```

and finally:

```cpp
std::optional<AdmissionCandidate> plan =
    program.inspect_admission(...);

if (!plan) {
    continue;
}
```

At present, a potentially useful candidate can disappear at any of these points without leaving enough information in the request log to explain why.

### `ResourceManager::Candidate`

Current private structure:

```cpp
struct Candidate {
    std::optional<AdmissionCandidate> plan;
    bool current_session_binding = false;
    std::optional<CatalogCapability> private_source;
    std::optional<CatalogCapability> shared_source;
    std::optional<PolicyObservationKey> selected_observation;
    std::optional<PrefixShortlistKey> source_key;
};
```

This is the ideal place to attach lightweight debug metadata for each admission candidate.

### `ResourceManager::plan_materialization()`

This translates `Candidate` objects into:

```cpp
MaterializationPlanner::CandidateInput
```

and constructs:

```text
MaterializationOwnerPolicy
MaterializationCheckpointPolicy
```

for the pressure search.

It also owns the important callback:

```cpp
logical_goal(...)
```

which determines whether a candidate/target can actually be represented by a valid publication slot and victim set.

---

## 2. `MaterializationPlanner<ModelContract, SearchClock>`

Current path:

```text
src/runtime/engine/context_cache/materialization_planner.h
```

Relevant class:

```cpp
template <class ModelContract, class SearchClock = std::chrono::steady_clock>
class MaterializationPlanner
```

### Candidate identity assessment

At the start of `MaterializationPlanner::plan()` every candidate gets its identity assessment:

```cpp
const IdentityMaterializationAssessment& identity =
    input.candidate->identity_assessment();

const FoldedCost cost =
    fold_identity(input, identity, machine_cost);
```

This gives the earliest point where we can answer:

```text
Did the ~109k candidate reach the planner?
Was it identity-feasible?
Was it expandable?
How much prompt reuse did its physical assessment contain?
What was its lower bound?
```

Important fields already available through `IdentityMaterializationAssessment` and `MaterializationMachineWork` include:

```text
physical_status
source_mode
pressure_may_change_machine_work
expandable
projection_work

pressure_transfers
candidate_transfers
optimistic_candidate_transfers
remaining_prefill_work
reused_prompt_tokens
```

### `fold_identity()`

This produces the initial `FoldedCost`:

```text
now_ns
total_ns
lower_bound_ns
copy_operations
transferred_bytes
remaining_text_prefill
remaining_vision_prefill
reused_prompt_tokens
current_session_binding
candidate_ordinal
```

These values should be exposed for diagnostics.

### Pressure search

If identity planning is not sufficient, the planner enters pressure planning through the model contract:

```cpp
program.begin_pressure_planning(...)
```

The planner then works with:

```text
PressureTargetGuidance
PressureTargetAssessment
PressureOwnerOutcome
PressureCheckpointRecoveryImpact
```

### `fold_guidance()`

This combines estimated target machine work with owner/checkpoint policy and `ContextPortfolioValue`.

Useful diagnostics include:

```text
estimated_immediate_ns
estimated_total_ns
degradation_units
owner_evictions
checkpoint_drops
affected_selected_hits
newest_affected_hit_epoch
retention_weight
explicit_shared_losses
remaining_prefill
reused_prompt_tokens
```

### `fold_assessment()`

This is the exact assessed-target pricing path.

It computes:

```text
now_ns
future_loss_ns
total_ns
lower_bound_ns
```

where `future_loss_ns` includes portfolio damage relative to retained checkpoints.

For this issue, that distinction is essential.

A good reuse target may have a small immediate cost but still lose if the planner assigns it a very large future portfolio loss.

That must be visible in the trace.

### `assess_target()`

This is where an exact pressure target is assessed and can replace the incumbent:

```cpp
if (goal && cost.less(incumbent.cost)) {
    incumbent = ...
}
```

This is the critical observation point for:

```text
candidate existed
-> target was assessed
-> target was feasible
-> exact cost was X
-> target did/did not beat incumbent Y
```

---

## 3. `MaterializationSearchBudget`

Current path:

```text
src/runtime/engine/context_cache/materialization_budget.h
```

Relevant class:

```cpp
class MaterializationSearchBudget
```

The initial search grant is currently:

```cpp
granted_(std::min(
    {std::uint64_t{5'000'000},
     economic(initial_cost),
     allowance.remaining(started)}))
```

So pressure search still begins with a maximum initial grant of 5 ms.

More importantly, after that initial grant the search can stop with:

```text
stop_reason = insufficient_expected_gain
budget_exhausted = false
```

For example:

```cpp
if (completion_ns > remaining || completion_ns > economic(gain_ns)) {
    reason_ = MaterializationStopReason::InsufficientExpectedGain;
    return false;
}
```

Other `InsufficientExpectedGain` paths include incomplete discovery restrictions and renewal-progress checks.

Therefore this combination:

```text
search_elapsed_ns ~= 5,000,000
search_granted_ns = 5,000,000
search_renewals = 0
stop_reason = insufficient_expected_gain
budget_exhausted = false
```

does **not** rule out the 5-ms search window as a contributing factor.

This is particularly relevant to Request #9:

```text
search_elapsed_ns = 4,999,026
search_granted_ns = 5,000,000
search_renewals = 0
stop_reason = insufficient_expected_gain
```

The planner appears to consume essentially the entire initial grant and then stop without a renewal.

What is not known is whether the good reuse candidate had already been assessed by then.

---

## 4. `ContextPortfolioValue`

Current path:

```text
src/runtime/engine/context_cache/context_portfolio_value.h
```

`fold_guidance()` and `fold_assessment()` use the portfolio model to account for the future value lost by evicting/degrading resident checkpoints.

Relevant inputs are built from:

```text
MaterializationOwnerPolicy
MaterializationCheckpointPolicy
```

with fields such as:

```text
retention_class
selected_hit_count
last_hit_epoch
private_retention_weight
explicit_shared_credit

demand_mask
rebuild_ns
baseline_recovery_ns
```

If the ~109k candidate is present and feasible but loses because of `future_loss_ns`, this layer becomes a primary suspect.

---

## 5. Context machine cost model

Current paths:

```text
src/runtime/engine/context_cache/context_cost.h
src/runtime/engine/context_cache/context_cost.cpp
```

Important pricing functions include:

```cpp
ContextMachineCostModel::prefill_ns(...)
ContextMachineCostModel::transfer_ns(...)
price_materialization_machine_work(...)
price_checkpoint_recovery_work(...)
```

`price_materialization_machine_work()` produces:

```text
optimistic_request_ns
immediate_ns
transferred_bytes
copy_operations
```

For the failing request, it is important to separate:

```text
predicted prefill cost
predicted transfer cost
predicted future portfolio loss
```

rather than only logging one aggregate total.

---

# Important diagnostic subtlety: `initial_predicted_total_ns`

`initial_predicted_total_ns` should not be interpreted as "the root candidate cost" without additional context.

Current `MaterializationPlanner::plan()` first evaluates all candidate identity assessments and chooses the best identity-feasible incumbent if one exists.

Only if no identity candidate can become the initial incumbent does the planner construct:

```cpp
session.root_maximal_target(...)
```

as the mandatory fallback.

Therefore this value:

```text
initial_predicted_total_ns = 833423220301
```

does not, by itself, tell us which candidate or target produced the ~833 s estimate.

The diagnostics need to include:

```text
initial candidate id
initial reuse path
initial reused tokens
initial target ordinal
initial root_maximal
initial degradation units
```

otherwise `initial_predicted_total_ns` remains ambiguous.

---

# Proposed diagnostic instrumentation

The goal is to make one failing request self-explanatory without dumping the entire pressure-search graph.

The instrumentation should be lightweight enough that it does not materially perturb a search that currently runs for only ~5 ms.

## A. Discovery summary

Add a request-level summary such as:

```json
"discovery": {
  "prefix_index_entries": 12,
  "valid_prefix_index_entries": 10,
  "shortlist_matches": 2,
  "inspect_admission_accepted": 2,
  "best_reuse_prompt_tokens": 109169
}
```

At minimum:

```text
prefix_index_entries
valid_prefix_index_entries
shortlist_matches
accepted_reuse_candidates
best_reuse_prompt_tokens
```

`best_reuse_prompt_tokens` is especially useful because it immediately separates:

```text
candidate discovery failure
```

from:

```text
candidate existed but planning/selection lost it
```

---

## B. Per-candidate discovery trace

For each root/private/shared admission candidate:

```json
{
  "candidate_id": 1,
  "kind": "private",
  "reuse_path": "private_endpoint",
  "reused_prompt_tokens": 109169,
  "current_session_binding": true,

  "source": {
    "slot": 1,
    "owner_id": 42,
    "revision": 17,
    "checkpoint_kind": "session_endpoint",
    "checkpoint_frontier": 109169,
    "checkpoint_ordinal": 0
  }
}
```

For candidates that are filtered before admission, record a compact rejection reason:

```text
invalid_prefix_index_entry
shortlist_key_mismatch
private_not_catalogued
private_handle_missing
private_active_edge
shared_not_catalogued
shared_handle_missing
shared_transaction_pinned
shared_active_edge
inspect_admission_rejected
zero_reusable_prompt
invalid_source_mode
```

This should be counters by default, with detailed entries only for the top few potentially useful prefix matches.

---

## C. Identity-cost trace

For each candidate that reaches `MaterializationPlanner`:

```json
{
  "candidate_id": 1,

  "identity": {
    "physical_status": "infeasible",
    "source_mode": "consume_to_active",
    "expandable": true,
    "pressure_may_change_machine_work": true,

    "reused_prompt_tokens": 109169,
    "remaining_prefill_tokens": 2089,

    "predicted_now_ns": 123456789,
    "predicted_total_ns": 123456789,
    "lower_bound_ns": 120000000,

    "transferred_bytes": 12345678,
    "copy_operations": 24,

    "logical_goal_available": false,
    "candidate_seeded": false
  }
}
```

This is the first decisive checkpoint.

If Request #9 shows:

```text
private_endpoint candidate
reused_prompt_tokens ~= 109169
```

here, discovery is working.

---

## D. Pressure-search summary per admission candidate

Do not log every pressure target.

Instead aggregate per candidate:

```json
{
  "candidate_id": 1,
  "pressure": {
    "targets_discovered": 8,
    "targets_assessed": 3,
    "feasible_targets": 1,

    "best_assessed_target": {
      "target_ordinal": 147,
      "source_mode": "consume_to_active",
      "degradation_units": 1,
      "root_maximal": false,

      "reused_prompt_tokens": 109169,
      "remaining_prefill_tokens": 2089,

      "predicted_now_ns": 1500000000,
      "predicted_future_loss_ns": 2200000000,
      "predicted_total_ns": 3700000000,
      "lower_bound_ns": 1400000000,

      "owner_evictions": 1,
      "checkpoint_drops": 1,
      "affected_selected_hits": 3,
      "transferred_bytes": 67108864,
      "copy_operations": 16
    }
  }
}
```

For the root candidate, emit exactly the same fields.

That makes root and reuse directly comparable.

---

## E. Initial and selected incumbent identity

Add:

```json
"initial_incumbent": {
  "candidate_id": 0,
  "target_ordinal": 123,
  "reuse_path": "root",
  "reused_prompt_tokens": 0,
  "degradation_units": 18,
  "root_maximal": true,
  "predicted_now_ns": 123,
  "predicted_future_loss_ns": 456,
  "predicted_total_ns": 579
}
```

and:

```json
"selected_incumbent": {
  "candidate_id": 0,
  "target_ordinal": 147,
  "reuse_path": "root",
  "reused_prompt_tokens": 0,
  "degradation_units": 1,
  "root_maximal": false,
  "predicted_now_ns": 123,
  "predicted_future_loss_ns": 456,
  "predicted_total_ns": 579
}
```

This removes the ambiguity in the existing:

```text
initial_predicted_total_ns
predicted_total_ns
selected_degradation_units
selected_maximal_fallback
```

fields.

---

## F. Search-budget refusal reason

`MaterializationStopReason::InsufficientExpectedGain` currently combines several materially different decisions.

For diagnostics, distinguish at least:

```text
completion_exceeds_remaining_allowance
completion_exceeds_economic_gain
discovery_not_eligible
discovery_already_used
no_progress_since_renewal
```

For the final rejected extension/renewal attempt record:

```json
"budget_decision": {
  "reason": "completion_exceeds_economic_gain",

  "elapsed_ns": 4999026,
  "granted_ns": 5000000,
  "remaining_allowance_ns": 44900000,

  "next_operation_ns": 20000,
  "completion_ns": 340000,
  "gain_ns": 5000000,
  "economic_gain_budget_ns": 250000,

  "complete_prediction": false,
  "discovery_eligible": true,
  "discovery_used": false,

  "progress": 36,
  "renewal_progress": 0
}
```

This is important because:

```text
budget_exhausted = false
```

currently does not tell us whether search effectively stopped at the initial 5-ms window.

---

# Suggested compact request-log schema

A failing request should ideally contain something similar to:

```json
{
  "result": {
    "prompt_tokens": 111258,
    "computed_prefill_tokens": 111258,
    "prefix_cache_hit_tokens": 0,
    "prefix_reuse_path": "root"
  },

  "materialization": {
    "stop_reason": "insufficient_expected_gain",
    "search_elapsed_ns": 4999026,
    "search_granted_ns": 5000000,
    "search_renewals": 0,

    "discovery": {
      "prefix_index_entries": 12,
      "shortlist_matches": 2,
      "accepted_reuse_candidates": 2,
      "best_reuse_prompt_tokens": 109169
    },

    "initial_incumbent": {
      "candidate_id": 0,
      "reuse_path": "root",
      "reused_prompt_tokens": 0,
      "root_maximal": true,
      "predicted_total_ns": 833423220301
    },

    "selected_incumbent": {
      "candidate_id": 0,
      "reuse_path": "root",
      "reused_prompt_tokens": 0,
      "degradation_units": 1,
      "root_maximal": false,
      "predicted_total_ns": 79156901670
    },

    "candidates": [
      {
        "candidate_id": 0,
        "reuse_path": "root",
        "identity_reused_prompt_tokens": 0,
        "identity_remaining_prefill_tokens": 111258,
        "targets_assessed": 20
      },
      {
        "candidate_id": 1,
        "reuse_path": "private_endpoint",
        "identity_reused_prompt_tokens": 109169,
        "identity_remaining_prefill_tokens": 2089,
        "targets_assessed": 0
      }
    ],

    "budget_decision": {
      "reason": "completion_exceeds_economic_gain"
    }
  }
}
```

Even this compact form would already answer most of the important questions.

---

# What each possible result would mean

| Trace result | Likely subsystem |
|---|---|
| `best_reuse_prompt_tokens = 0` | candidate discovery / prefix index / residency / shortlist matching |
| ~109k candidate exists but never reaches planner | `inspect_admission()` / ResourceManager filtering |
| ~109k candidate reaches planner but `targets_assessed = 0` | search ordering / initial 5-ms grant / economic renewal |
| ~109k target assessed but never feasible | physical pressure planning / reservation geometry |
| ~109k target feasible, immediate cost low, `future_loss_ns` huge | demand / retention / `ContextPortfolioValue` |
| ~109k target feasible but `now_ns` huge | transfer / prefill cost model |
| ~109k target feasible and cheaper than root but root selected | planner ordering / incumbent update / sealing inconsistency |
| ~109k target selected internally but final result reports root | post-planner materialization / result propagation |

---

# Related upstream issues

## #176 — Materialization search budget capped at 5 ms

[Issue #176](https://github.com/Neroued/ninfer/issues/176)

Title:

> Materialization search budget is capped at 5 ms, so the search terminates on time_budget for any incumbent above 100 ms

Status:

```text
closed
```

Why it is relevant:

- established that a flat 5-ms pressure-search ceiling can prevent the planner from reaching reusable targets;
- showed long-context root fallbacks that disappear when the search can complete;
- explicitly noted that fixing the ceiling did not solve every warm-catalog reuse failure.

The present reproduction differs because the current stop reason is:

```text
insufficient_expected_gain
```

rather than:

```text
time_budget
```

However, current `MaterializationSearchBudget` still begins with a maximum 5-ms grant, and a failed renewal can produce `InsufficientExpectedGain` while leaving `budget_exhausted = false`.

So #176 remains directly relevant to the search-budget part of this investigation.

---

## #177 — private continuation saturation

[Issue #177](https://github.com/Neroued/ninfer/issues/177)

Title:

> Private continuation cache can become permanently saturated across independent conversations

Status:

```text
open
```

Why it is related:

- concerns pressure planning and private-continuation reclamation;
- documents a case where physical pressure and the value model interact badly;
- demonstrates a demand/value chicken-and-egg problem.

Why it is not the same reproduction:

- #177 focuses on independent conversations and saturation across them;
- this issue is one monotonically growing conversation that already demonstrated successful reuse immediately before the failure.

---

## #180 — rolling context-cache retention

[Issue #180](https://github.com/Neroued/ninfer/issues/180)

Title:

> Proposal: opt-in rolling context-cache retention for long agent sessions

Status:

```text
open
```

Why it is related:

- addresses long append-only conversations;
- changes how descendant captures inherit proven ancestor demand;
- directly targets lineage progression under retention pressure.

Why it is not sufficient here:

- rolling retention was ported to the current tree;
- it was enabled and verified in the running process;
- the warm -> root collapse still reproduced with another model.

Therefore the current issue should not be treated primarily as a rolling-retention bug.

---

## #229 — root re-prefill caused by bounded materialization search

[Issue #229](https://github.com/Neroued/ninfer/issues/229)

Title:

> [Bug] Materialization planner: fixed 5ms pressure-search budget causes root re-prefill fallback for large requests in multi-session states

Status:

```text
open
```

This is currently the closest public issue.

It reports:

```text
best_reuse_prompt_tokens > 0
root selected
short pressure search
full re-prefill
```

and demonstrates that the reuse candidate can exist even though root ultimately wins.

The current reproduction differs in two important ways:

1. it is a single monotonically growing conversation rather than a multi-session workload;
2. current diagnostics report:

```text
stop_reason = insufficient_expected_gain
budget_exhausted = false
selected_maximal_fallback = false
```

instead of the `time_budget` / maximal-fallback combination shown in #229.

The instrumentation proposed here should make it possible to determine whether this is another manifestation of the same underlying search limitation or a separate cost/portfolio-selection failure.

---

## #251 — prefix reuse stops after checkpoint budget pressure

[Issue #251](https://github.com/Neroued/ninfer/issues/251)

Title:

> Prefix reuse stops once the checkpoint budget is exercised, and only an engine restart restores it (no LRU eviction observed)

Status:

```text
open
```

Why it is related:

- long-lived cache state can transition from healthy reuse to persistent root behavior;
- checkpoint / continuation budgets are involved;
- current external logging is not sufficient to explain why reuse disappears.

Why it is not necessarily the same bug:

- #251 describes a broader persistent degradation after checkpoint-budget pressure;
- the reproduction here captures a precise adjacent-turn transition where a ~109k hit is followed immediately by zero reuse.

---

# What this issue is NOT claiming

This issue is intentionally narrower than a general "context cache is broken" report.

It does **not** currently claim that:

```text
rolling retention is wrong
the 5-ms search grant is definitely the sole cause
the portfolio value model is definitely wrong
the cost model is definitely miscalibrated
the candidate is definitely still resident
```

Those are exactly the possibilities that the missing diagnostics need to distinguish.

The only currently demonstrated facts are:

```text
1. A ~109k prefix is successfully reused on one request.
2. The immediately following append-only request can select root and reuse zero tokens.
3. Rolling retention does not prevent the failure.
4. The failing request goes through materialization search.
5. Its search consumes essentially the full initial 5-ms grant.
6. The final selected plan has one degradation unit.
7. Current diagnostics do not identify which admission candidate / target produced the initial and final predicted costs.
```

---

# Requested change

Please add enough structured materialization diagnostics to distinguish:

```text
candidate discovery
-> identity assessment
-> pressure-target discovery
-> pressure-target assessment
-> portfolio/cost fold
-> search-budget stop
-> incumbent selection
-> sealing/materialization
```

for one request.

The minimum useful addition would be:

```text
best_reuse_prompt_tokens
initial incumbent candidate + target
selected incumbent candidate + target
per-candidate identity summary
per-candidate targets assessed / feasible
best assessed target per candidate
exact search-budget refusal reason
```

No policy change is required to make progress on this bug.

---

# Acceptance criteria for the diagnostic patch

For a reproduction equivalent to Request #9, one request-log record should be sufficient to answer all of these questions:

1. Did a candidate corresponding to the previous ~109k prefix exist in the prefix index?
2. Did it pass shortlist-key matching?
3. Did `Program::inspect_admission()` accept it?
4. What `reused_prompt_tokens` did its identity assessment report?
5. Was identity materialization feasible?
6. If not, was it expandable under pressure?
7. How many pressure targets for that candidate were discovered?
8. How many were actually assessed before search stopped?
9. Was any ~109k target physically feasible?
10. What were its:
    - immediate cost,
    - future portfolio loss,
    - total cost,
    - degradation units,
    - transfer work,
    - remaining prefill?
11. What exact candidate/target was the initial incumbent?
12. What exact candidate/target was ultimately selected?
13. Why did the search stop at ~5 ms?
14. If the ~109k target lost to root, what exact comparison made root cheaper?
15. If the ~109k target was never assessed, what prevented the planner from reaching it?

Once those answers are available, the next fix should be straightforward to scope to the correct layer instead of changing retention policy speculatively.

---

## Short version

Current behavior:

```text
109545 prompt
-> 109169 cache hit
-> private_endpoint

next turn:

111258 prompt
-> 0 cache hit
-> root
```

The main missing information is:

```text
Was the 109169-token candidate still present?

YES:
    why did MaterializationPlanner reject / never assess / out-price it?

NO:
    where did ResourceManager / prefix discovery / residency lose it?
```

That distinction should be observable directly in `--request-log-jsonl`.
