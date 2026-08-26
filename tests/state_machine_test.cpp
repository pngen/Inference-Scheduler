#include "test_util.hpp"
int main() {
  using namespace ts;
  auto clock = std::make_shared<SimulatedClock>();
  Scheduler s(default_scheduler_config(), clock);
  s.register_worker(make_worker(WorkerId(1), WorkerBootId(1), 100));
  RequestId r(7200);
  auto ad = s.submit(make_spec(TenantId(1), ModelIdentity(10), RequestPhase::Prefill), r);
  auto d = s.plan_dispatch(r);
  auto snap = s.request_snapshot(r);
  CompletionReport start; start.epoch = s.epoch(); start.worker = d.value().worker_id; start.boot_id = WorkerBootId(1);
  start.request_id = r; start.attempt_id = snap.value().attempt_id; start.generation = snap.value().generation;
  start.worker_snapshot.available_capacity_units = 100;
  REQUIRE(s.confirm_dispatch_start(start).ok(), "confirm");
  CompletionReport comp; comp.epoch = s.epoch(); comp.worker = d.value().worker_id; comp.boot_id = WorkerBootId(1);
  comp.request_id = r; comp.attempt_id = snap.value().attempt_id; comp.generation = snap.value().generation;
  comp.status = CompletionStatus::Succeeded; comp.worker_snapshot.available_capacity_units = 100;
  auto co = s.complete_attempt(comp);
  REQUIRE(co.ok() && co.value().acceptance == CompletionAcceptance::Accepted, "accepted");

  // terminal request cannot be cancelled
  auto cx = s.cancel(r, CancellationReason::ClientRequested, "x");
  REQUIRE(!cx.ok(), "cancelling a completed request is invalid");
  REQUIRE(cx.error().code == ErrorCode::InvalidState, "invalid state code");

  // duplicate completion on terminal rejected
  auto co2 = s.complete_attempt(comp);
  REQUIRE(co2.ok() && co2.value().acceptance == CompletionAcceptance::Duplicate, "duplicate completion");

  // planning a request that is not runnable returns Hold
  auto d2 = s.plan_dispatch(r);
  REQUIRE(d2.ok() && d2.value().decision == DispatchDecision::Hold, "not runnable hold");
  REQUIRE(d2.ok() && d2.value().reason_code == "not_runnable", "not runnable reason");

  // unknown request
  auto d3 = s.plan_dispatch(RequestId(999999));
  REQUIRE(!d3.ok() && d3.error().code == ErrorCode::NotFound, "unknown request not found");

  std::printf("  OK state_machine scenario\n");
  return test_result("state_machine_test");
}
