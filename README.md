# Inference Scheduler

An open-source, vendor-neutral C++20 runtime for scheduling inference work across
heterogeneous AI serving infrastructure. Inference Scheduler owns the scheduling
decision and execution authority around inference work: admission, queueing,
batching, prefill/decode coordination, fairness, deadlines, cancellation, retries,
accelerator-aware dispatch, capacity reservation, and the distributed authority
that binds completions to a scheduler epoch, worker boot identity, request, and
attempt.

## The systems question

**What inference work should run next, where should it run, and under what
latency, fairness, capacity, batching, and execution constraints?**

Inference Scheduler answers that question. It does not implement a model server, a
tokenizer, a model loader, a KV cache, a tensor cache, a memory allocator, a
transport library, a generic job scheduler, or a CUDA kernel framework. External
systems may provide model residency, memory, topology, reusable state, transfer,
or execution capability. Inference Scheduler consumes explicit information about
those conditions and decides scheduling order and dispatch according to
scheduler-owned policy.

## Ownership boundary

The scheduler governs:

* admission / defer / reject / backpressure
* queue ordering and runnable-ness
* priority, latency classes, deadlines
* weighted fairness with starvation prevention
* batching eligibility, dynamic batch formation, seal/finalization
* prefill/decode phase coordination
* accelerator/node eligibility, capacity awareness, locality/warmth hints
* dispatch decisions, reservations/claims (no oversubscription)
* cancellation and retry eligibility, attempt identity, stale-attempt rejection
* completion authority and shutdown/drain behavior
* deterministic scheduling explanations, statistics, events
* persistence/recovery of scheduler-owned durable metadata

The scheduler is **not**: a model server, tokenizer, model loader, KV-cache
implementation, tensor cache, memory allocator, memory-pressure runtime, model
artifact cache, transport library, generic distributed job scheduler, CUDA kernel
framework, or a full inference engine. It is not a wrapper around vLLM, SGLang,
TensorRT-LLM, llama.cpp, PyTorch, Ray, Kubernetes, or Slurm.

## Request lifecycle

Requests move through an explicit state machine:

`Created -> PendingAdmission -> Admitted -> Queued -> Reserved -> Dispatched ->
Running -> Completing -> Completed`

Terminal/error states: `Rejected`, `Cancelled`, `Expired`, `Failed`. Transitions
are validated; illegal transitions fail deterministically; terminal requests can
never silently return to a runnable state. Retries create explicit execution
attempts (each with a new `AttemptId` and `Generation`) rather than rewinding a
single mutable request record. Cancellation races resolve at a documented commit
point: whichever of a cancellation or an accepted completion the scheduler
processes first commits; a completion that arrives after a committed cancellation
is discarded as a cancelled outcome.

## Scheduling semantics

### Admission

Bounded admission evaluates configurable constraints: maximum globally admitted
and queued requests, per-tenant concurrency and queue depth, token budget,
estimated memory, accelerator capability, model compatibility, deadline
feasibility, latency class, and queue pressure. The result is an explicit
`Admit` / `Defer` / `Reject` with a stable, machine-readable reason code.
Backpressure is first-class; queues are bounded.

### Queueing and fairness

Ready work is organized into weighted-fair flows keyed by (tenant, latency class,
priority, phase). A virtual-time weighted fair queue (WFQ) assigns each flow a
share proportional to the product of explicit, configurable weights (tenant,
latency, priority, phase). This gives bounded priority advantage (never a full
preemption of lower classes), starvation prevention, and correct accounting for
uneven request sizes via cost-weighted virtual finish times. Within a flow the
most urgent request (by absolute-deadline slack, then priority, then request id)
is served first. Neither a single flooded tenant nor a flooded phase can starve
another indefinitely. Per-flow virtual time is exposed in the snapshot for
introspection.

### Deadlines and latency classes

Deadlines are absolute instants on an injectable monotonic clock. Work whose
deadline has passed is never dispatched. Admission can reject or defer
deadline-infeasible work according to policy. The four latency classes
`interactive` / `standard` / `throughput` / `background` are configurable weights,
not hard-coded behavior. Queue delay, schedule delay, and dispatch delay are
tracked separately.

### Batching

Batch formation is scheduler-owned. Compatible requests (matching model,
revision, adapter, phase (when mixing is disabled), and latency class) combine
up to `max_batch_size` and a token cap. Incompatible requests never combine.
Batch membership is deterministic (sorted by deadline slack, priority, request
id). A `max_wait` bounds how long formation waits; cancellation is honored while
a batch is unsealed. Dispatch runs a sealed batch as a unit, and partial
per-request results are handled individually.

### Prefill / decode coordination

Inference Scheduler understands two phases, `PREFILL` and `DECODE`, and treats
them distinctly. Prefill is larger/coarser; decode is iterative, sensitive work.
A configurable phase weight gives decode more scheduling share by default, so
large prefill cannot starve decode, while a bounded weight also prevents decode
saturation from permanently blocking new prefill. Per-sequence decode progress is
accounted, decode steps can be cancelled between steps, and sequences complete
without orphaned scheduler state. Phases are not required to serialize: prefill
and decode may both be in flight.

### Worker and accelerator model

Workers advertise capability: identity (worker/node/accelerator), boot identity,
backend type, total/available capacity units, supported models/revisions/adapters
and phases, batch constraints, locality tag, and warmth hints. The scheduler never
pretends to know information it was not given.

### Accelerator-aware dispatch

Dispatch ranks eligible candidates with explicit, inspectable weighted factors:
capability match, available capacity, queue/load, expected cost, deadline
urgency, latency class, phase affinity, locality, model warmth, state warmth,
batching opportunity, and fairness. The chosen destination is reported with the
full score/component breakdown in an `ExplainResult`, so no decision hides behind
opaque magic constants. Ranking is deterministic with a stable tie-break.

### Capacity reservation and oversubscription safety

A dispatch follows: eligible candidates -> rank -> reserve capacity -> commit
authority -> hand off -> worker accepts -> running -> completion/release. Capacity
returns to baseline after success, failure, cancellation, retry exhaustion, and
shutdown. No leaked reservations, no negative capacity, and no silent
oversubscription.

## Distributed authority

A local distributed control plane uses framed TCP (bounded 4-byte length prefix,
16 MiB frame cap). Separate OS processes run the scheduler/coordinator, >= 2
workers, and a client/validation driver. Workers carry boot identities; the
scheduler carries an epoch. A completion binds exactly to (scheduler epoch,
worker boot id, request id, attempt id, generation). A stale worker from an old
epoch, a restarted worker with a new boot identity, a mismatched/null/double
attempt, or an impossible completion is rejected deterministically. Malformed or
oversized frames are rejected rather than allocated. Ambiguous state-changing
RPCs are not silently retried in a way that duplicates execution; idempotency is
deliberate.

## Retries, cancellation, backpressure

Failures are classified (retryable transport, retryable worker, non-retryable
request, scheduler rejection, deadline expiry, cancellation, stale authority,
ambiguous dispatch). A retryable failure creates a new attempt id only if retry
budget and deadline remain. Cancellation is real: while awaiting admission, queued,
unsealed-batch, post-reservation, in-flight, running, and between decode steps.
Backpressure is exposed as structured decisions/events (queue saturation,
admission saturation, worker saturation, per-tenant saturation, deadline
infeasibility, batch pressure), never sleeps or log strings.

## Persistence and recovery

Scheduler-owned durable metadata is persisted as a versioned binary envelope with
a checksum and atomic replace (`write temp -> FlushFileBuffers -> MoveFileEx`).
Recovery restores epoch lineage, durable request metadata, per-tenant/fairness
accounting, attempt history, and configuration/version metadata. Deserialization
is fully bounded and rejects bad magic, bad version, count overflows, impossible
enum values, truncation, corruption, and inconsistent state. External execution
that cannot be proven alive is never invented as successful; ambiguous pre-crash
in-flight work enters a defined recovery state.

## Threading model

The scheduler is thread-safe for concurrent submission, cancellation, worker
updates, dispatch, completion, snapshots, stats reads, and event consumption. A
single authoritative state mutex guards all scheduler state; event callbacks are
invoked outside any internal lock. The flow/worker/batch structures use indexed
lookups so queue, request, attempt, worker, cancellation, and completion paths do
not scan the entire scheduler state. There is no detached-thread ambiguity: the
coordinator joins/closes resources deterministically on shutdown.

## CUDA validation

A CUDA-backed executor performs real, bounded GPU work (a prefill-like kernel and
a decode-like repeated-kernel schedule) on the installed NVIDIA accelerator. It
detects the device, compute capability, and memory; allocates bounded device
memory; launches kernels; synchronizes; and verifies a host-side checksum. It
performs no LLM inference and accesses no model weights. The CUDA validation test
runs it against the installed GPU (e.g. an RTX 5090 / sm_120).

## Building

Requirements: C++20 (Visual Studio 2022 MSVC), CMake 3.20+, Ninja, and (for the
CUDA backend) CUDA 12.8+ with a compatible driver.

```powershell
scripts\build.ps1 -Configuration Release            # CPU build
scripts\build.ps1 -Configuration Release -Cuda      # CUDA backend build
scripts\build.ps1 -Configuration Debug              # debug build
```

## Tests

All tests are registered with CTest and run to completion:

```powershell
ctest --test-dir build --output-on-failure
```

The suite covers: state machine, admission, queueing, fairness, deadlines,
batching, prefill/decode, dispatch/reservation, cancellation, retry, persistence
and corruption rejection, stale authority, concurrency (randomized property),
networking, and the multiprocess proof. A seeded property test runs thousands of
randomized submit/admit/queue/dispatch/cancel/fail/retry/complete/restart
operations and asserts invariants after each step.

## CLI

`inference_scheduler <command>`

`serve` (coordinator), `worker`, `submit`, `cancel`, `status`, `queue`, `workers`,
`stats`, `snapshot`, `explain`, `drain`, `recover`, `bench`. The `explain` command
shows why a request was admitted/rejected/deferred and why a worker was selected or
rejected.

## Installation and downstream usage

```powershell
cmake --install build --prefix _install
```

Then a downstream project can `find_package(InferenceScheduler CONFIG REQUIRED)`,
link `InferenceScheduler::InferenceScheduler`, include
`<inference_scheduler/inference_scheduler.hpp>`, and use the API.

## Examples

Runnable examples live in `examples/` and cover basic admission, fairness,
dynamic batching, prefill/decode, cancellation and retry, and explain_decision.

## Benchmark methodology

`scheduler_bench` measures **completed** work only. Submission throughput counts
requests admitted; completion throughput counts requests that reached `Completed`;
batch_shrink is average members per formed batch (batching benefit); and
plan_dispatch latency is the median single-decision cost. No asynchronous
submission timing is reported as throughput while work remains unfinished.

## License

Apache License 2.0. Copyright 2026 Summon Software Labs. No telemetry transmission.
