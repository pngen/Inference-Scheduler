#include "test_util.hpp"
int main() {
  using namespace ts;
  auto clock = std::make_shared<SimulatedClock>();
  Scheduler s(default_scheduler_config(), clock);
  s.register_worker(make_worker(WorkerId(1), WorkerBootId(1), 100));
  RequestId r(7100);
  auto ad = s.submit(make_spec(TenantId(1), ModelIdentity(10), RequestPhase::Prefill), r);
  auto d = s.plan_dispatch(r);
  REQUIRE(d.ok() && d.value().decision == DispatchDecision::Dispatch, "dispatch");
  auto snap = s.request_snapshot(r);

  // (1) stale epoch
  CompletionReport old_epoch; old_epoch.epoch = SchedulerEpoch(999); old_epoch.worker = d.value().worker_id; old_epoch.boot_id = WorkerBootId(1);
  old_epoch.request_id = r; old_epoch.attempt_id = snap.value().attempt_id; old_epoch.generation = snap.value().generation; old_epoch.status = CompletionStatus::Succeeded;
  old_epoch.worker_snapshot.available_capacity_units = 100;
  auto co1 = s.complete_attempt(old_epoch);
  REQUIRE(co1.ok() && co1.value().acceptance == CompletionAcceptance::StaleEpoch, "stale epoch");

  // (2) worker restart with new boot id -> old boot authority rejected
  // re-register same worker id with a new boot id
  auto w = make_worker(WorkerId(1), WorkerBootId(77777), 100);
  s.register_worker(w);
  CompletionReport old_boot; old_boot.epoch = s.epoch(); old_boot.worker = WorkerId(1); old_boot.boot_id = WorkerBootId(1);
  old_boot.request_id = r; old_boot.attempt_id = snap.value().attempt_id; old_boot.generation = snap.value().generation; old_boot.status = CompletionStatus::Succeeded;
  old_boot.worker_snapshot.available_capacity_units = 100;
  auto co2 = s.complete_attempt(old_boot);
  REQUIRE(co2.ok() && co2.value().acceptance == CompletionAcceptance::StaleWorker, "stale worker boot id");

  // stale attempt id (generation must also match)
  CompletionReport bad_gen; bad_gen.epoch = s.epoch(); bad_gen.worker = WorkerId(1); bad_gen.boot_id = WorkerBootId(77777);
  bad_gen.request_id = r; bad_gen.attempt_id = snap.value().attempt_id; bad_gen.generation = Generation(424242); bad_gen.status = CompletionStatus::Succeeded;
  bad_gen.worker_snapshot.available_capacity_units = 100;
  auto co3 = s.complete_attempt(bad_gen);
  REQUIRE(co3.ok() && co3.value().acceptance == CompletionAcceptance::StaleAttempt, "stale generation/attempt");

  return test_result("stale_test");
}
