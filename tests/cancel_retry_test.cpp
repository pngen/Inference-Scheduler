#include "test_util.hpp"
int main() {
  using namespace ts;
  auto clock = std::make_shared<SimulatedClock>();
  auto cfg = default_scheduler_config();
  Scheduler s(cfg, clock);
  auto w1 = make_worker(WorkerId(1), WorkerBootId(1), 100);
  s.register_worker(w1);

  // (1) queued cancellation is authoritative
  RequestId q(6000);
  auto aq = s.submit(make_spec(TenantId(1), ModelIdentity(10), RequestPhase::Prefill), q);
  REQUIRE(aq.ok(), "queued submit");
  auto cr = s.cancel(q, CancellationReason::ClientRequested, "user");
  REQUIRE(cr.ok(), "queued cancel ok");
  auto sq = s.request_snapshot(q);
  REQUIRE(sq.ok() && sq.value().state == RequestState::Cancelled, "queued request cancelled");

  // (2) in-flight cancellation: success after cancel is rejected as cancelled
  auto ex = std::make_shared<CpuExecutor>(2000, 400000);
  LocalDriver drv(s, ex, {w1});
  RequestId r(6001);
  auto ar = s.submit(make_spec(TenantId(2), ModelIdentity(20), RequestPhase::Prefill), r);
  auto dr = s.plan_dispatch(r);
  REQUIRE(dr.ok() && dr.value().decision == DispatchDecision::Dispatch, "dispatch ok");
  auto snap = s.request_snapshot(r);
  CompletionReport start;
  start.epoch = s.epoch(); start.worker = dr.value().worker_id; start.boot_id = w1.boot_id;
  start.request_id = r; start.attempt_id = snap.value().attempt_id; start.generation = snap.value().generation;
  start.worker_snapshot.available_capacity_units = 100;
  REQUIRE(s.confirm_dispatch_start(start).ok(), "confirm start");
  // cancel while running
  auto cr2 = s.cancel(r, CancellationReason::ClientRequested, "inflight");
  REQUIRE(cr2.ok(), "inflight cancel ok");
  // late success completion must be treated as cancelled outcome
  CompletionReport comp; comp.epoch = s.epoch(); comp.worker = dr.value().worker_id; comp.boot_id = w1.boot_id;
  comp.request_id = r; comp.attempt_id = snap.value().attempt_id; comp.generation = snap.value().generation;
  comp.status = CompletionStatus::Succeeded;
  comp.worker_snapshot.available_capacity_units = 100;
  auto co = s.complete_attempt(comp);
  REQUIRE(co.ok() && co.value().acceptance == CompletionAcceptance::CancelledOutcome, "cancelled outcome after cancel");
  auto sr = s.request_snapshot(r);
  REQUIRE(sr.ok() && sr.value().state == RequestState::Cancelled, "request ends Cancelled");
  REQUIRE(s.stats().requests_cancelled >= 2, "cancelled stat");

  // (3) retry: retryable failure -> new attempt; old attempt becomes stale
  RequestId rr(6002);
  auto arr = s.submit(make_spec(TenantId(3), ModelIdentity(30), RequestPhase::Prefill), rr);
  auto drr = s.plan_dispatch(rr);
  REQUIRE(drr.ok() && drr.value().decision == DispatchDecision::Dispatch, "retry dispatch");
  auto snapr = s.request_snapshot(rr);
  CompletionReport start2; start2.epoch = s.epoch(); start2.worker = drr.value().worker_id; start2.boot_id = w1.boot_id;
  start2.request_id = rr; start2.attempt_id = snapr.value().attempt_id; start2.generation = snapr.value().generation;
  start2.worker_snapshot.available_capacity_units = 100;
  REQUIRE(s.confirm_dispatch_start(start2).ok(), "retry confirm");
  CompletionReport fail; fail.epoch = s.epoch(); fail.worker = drr.value().worker_id; fail.boot_id = w1.boot_id;
  fail.request_id = rr; fail.attempt_id = snapr.value().attempt_id; fail.generation = snapr.value().generation;
  fail.status = CompletionStatus::Failed; fail.failure_class = FailureClass::RetryableWorker; fail.error_message = "boom";
  fail.worker_snapshot.available_capacity_units = 100;
  auto cf = s.complete_attempt(fail);
  REQUIRE(cf.ok() && cf.value().retried, "retried flag set");
  auto after_retry = s.request_snapshot(rr);
  REQUIRE(after_retry.ok() && after_retry.value().state == RequestState::Queued, "re-queued after retry");
  REQUIRE(after_retry.ok() && after_retry.value().attempt_id != snapr.value().attempt_id, "new attempt id after retry");

  // stale old attempt completion rejected
  CompletionReport old; old.epoch = s.epoch(); old.worker = drr.value().worker_id; old.boot_id = w1.boot_id;
  old.request_id = rr; old.attempt_id = snapr.value().attempt_id; old.generation = snapr.value().generation;
  old.status = CompletionStatus::Succeeded; old.worker_snapshot.available_capacity_units = 100;
  auto co_old = s.complete_attempt(old);
  REQUIRE(co_old.ok() && co_old.value().acceptance == CompletionAcceptance::StaleAttempt, "old attempt stale");
  return test_result("cancel_retry_test");
}
