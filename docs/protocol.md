# Protocol

Inference Scheduler's distributed authority uses a length-prefixed framed TCP
protocol. Every socket message is a JSON object. A frame is a 4-byte little-endian
length prefix followed by the payload. The frame length is bounded (16 MiB); a
zero or oversized length rejects the frame deterministically.

## Processes

* **coordinator** hosts the `Scheduler` and listens for worker and client
  connections.
* **worker** connects, registers, then serves `run` requests by executing the
  CPU or CUDA executor and reporting per-member results.
* **client** is the validation driver: it waits for workers, submits requests,
  drives dispatch, and inspects statistics.

## Message vocabulary

### Worker to coordinator

* `{"op":"register","worker":W,"boot":B,"node":N,"accel":A,"units":U,"backend":"cpu|
  "cuda","device":"...","models":[...]}`
* `{"op":"completed","batch_id":ID,"units":U,"results":[{"req","attempt","gen","status",
  "failure","out","work","dur","msg"}]}`

### Coordinator to worker

* `{"op":"run","batch_id":ID,"members":[{"req","attempt","gen","phase","model","rev",
  "adapter","tin","tout","cost","payload"}]}`

### Client to coordinator

* `{"op":"ready"}` -> `{"op":"ready_ack","workers":N}`
* `{"op":"submit",...spec...}` -> `{"op":"submit_ack","decision","reason","request_id",
  "attempt","generation"}` (ids encoded as decimal strings to preserve 64-bit precision)
* `{"op":"dispatch","request_id":R,"explain":bool}` ->
  `{"op":"dispatch_ack","decision","worker","batch","reason","members":[...]}`
* `{"op":"cancel","request_id":R}` -> `{"op":"cancel_ack","ok","reason"}`
* `{"op":"stats"}` -> `{"op":"stats_ack",...aggregate counters...}`
* `{"op":"shutdown"}`

## Authority binding

A completion is accepted only when all of the following match the scheduler's
authoritative state: scheduler epoch, worker boot id (bound to the registered
worker), request id, attempt id, and generation. A stale epoch, a restarted worker
bearing a new boot id, a mismatched/null/double attempt, or an impossible
completion is rejected deterministically and counted in `stale_rejected`.

## Idempotency

Ambiguous state-changing operations are not silently retried. A re-queue on
recovery creates a new attempt identity, so a late completion from an old attempt
is rejected as stale rather than accepted as a duplicate success. Existing
terminal requests reject duplicate completions.
