# Architecture

Inference Scheduler is a self-contained runtime. The core `Scheduler` is a pure
decision engine (no I/O, injectable monotonic clock); a `LocalDriver` wires it to
an in-process executor; the distributed control plane wires it to workers over
framed TCP.

## Component layers

* **Public API** (`include/inference_scheduler/`): strong types (`Id<Tag>`, enums,
  `Result<T>`/`Error`), policy/config structs, the `Scheduler` class, an `Executor`
  interface, the `CpuExecutor`, and the `LocalDriver`.
* **Scheduler core** (`src/scheduler.cpp`): the authoritative state machine,
  admission, WFQ queueing, dynamic batching, prefill/decode weighting, dispatch
  scoring, reservation, cancellation, retry, persist/recover.
* **Persistence** (`src/persistence.cpp`): versioned binary envelope, checksum,
  atomic replace, bounded, corruption-rejecting deserialization.
* **Net** (`src/net.*`, `src/protocol.hpp`): framed TCP + a bounded JSON message
  vocabulary for the coordinator/worker/client processes.
* **Apps** (`src/*_app.cpp`, `src/cli.cpp`): the coordinator, worker, client, and
  CLI w.r.t. the runtime.

## Internal state and data structures

All authoritative state lives behind one mutex in `Scheduler::Impl`: `requests_`
(request + attempt history), `workers_`, `tenants_` (fairness + accounting),
`flows_` (per-flow WFQ, keyed by tenant/latency/priority/phase), an active-flow
set, a compatibility index (`compat_`), and a `queued_` set. Global counters are
`global_admitted_`, `global_queued_`, `global_reserved_units_`, and
`global_running_units_`.

* Requests are never scanned for lookup: `requests_` is keyed by `RequestId`.
* Queueing uses per-flow heaps (deadline-ordered; lazy deletion by
  generation/queued check on pop), and a WFQ selector over active flows.
* Batch compatibility is indexed by `CompatKey = (model, revision, adapter,
  phase, latency)`, so batch formation narrows candidates instead of scanning the
  whole queue.
* Reservations are per-(worker, batch); capacity equates to `available_units -
  reserved_units` and is guarded against going negative.

## State machine

Request states: `Created -> PendingAdmission -> Admitted -> Queued -> Reserved ->
Dispatched -> Running -> Completing -> Completed`, with terminal `Rejected /
Cancelled / Expired / Failed`. Attempts go `Created -> Reserved -> Dispatched ->
Running -> Completing -> Succeeded / Failed / Cancelled / Stale / Expired`. Each
transition is validated; terminal states are irreversible.

## Determinism

Selection is deterministic given equivalent state, inputs, policy, and clock:
WFQ uses explicit weights; within a flow the ordering is by absolute-deadline
slack, then priority, then request id; worker tie-breaks use worker id; batch
membership is sorted. A `SimulatedClock` drives unit tests; production uses the
steady clock. ID salts are configurable for reproducible ID sequences.

## Threading and lock discipline

One mutex guards the authoritative scheduler state. Event emission appends to a
bounded log under a separate event mutex; handlers run after release of the main
lock. The coordinator is multi-threaded (per connection); each worker socket is
read by one thread; dispatch is synchronized per worker slot with a condition
variable. No callback runs under the scheduler lock. No thread owns a socket that
another thread also reads.

## Failure accounting

Capacity returns to baseline on success, failure, cancellation, retry, and
shutdown. Per-batch pending counters drive batch release; in-flight requests on a
disappeared/restarted worker are recovered (re-queued as new attempts or failed)
and reservations are released.
