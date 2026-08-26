#include "test_util.hpp"
int main() {
  using namespace ts;
  auto clock = std::make_shared<SimulatedClock>();
  auto cfg = default_scheduler_config();
  Scheduler s(cfg, clock);
  s.register_worker(make_worker(WorkerId(1), WorkerBootId(1), 100));

  RequestId rid(5000);
  auto spec = make_spec(TenantId(1), ModelIdentity(10), RequestPhase::Prefill);
  spec.deadline = Deadline::after(std::chrono::milliseconds(40), clock->now());
  auto ad = s.submit(spec, rid);
  REQUIRE(ad.ok() && ad.value().decision == AdmissionDecision::Admit, "admitted with deadline");

  // Advance the simulated clock so the deadline passes while queued.
  clock->advance(std::chrono::milliseconds(100));
  auto d = s.plan_dispatch(rid, false);
  REQUIRE(d.ok(), "plan ok");
  REQUIRE(d.value().decision == DispatchDecision::Hold, "expired request not dispatched");
  REQUIRE(d.value().reason_code == "deadline_expired", "expired reason");
  auto snap = s.request_snapshot(rid);
  REQUIRE(snap.ok() && snap.value().state == RequestState::Expired, "request expired");

  // Submission of an already-expired deadline is rejected at admission.
  RequestId rid2(5001);
  auto spec2 = make_spec(TenantId(2), ModelIdentity(10), RequestPhase::Prefill);
  spec2.deadline = Deadline::after(std::chrono::milliseconds(-5), clock->now());
  clock->advance(std::chrono::milliseconds(10));
  auto ad2 = s.submit(spec2, rid2);
  REQUIRE(ad2.ok() && ad2.value().decision == AdmissionDecision::Reject, "already-expired rejected at submit");
  return test_result("deadline_test");
}
