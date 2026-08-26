#include "test_util.hpp"
int main() {
  using namespace ts;
  auto clock = std::make_shared<SimulatedClock>();

  // global admission cap
  auto cfg = default_scheduler_config();
  cfg.admission.max_global_admitted = 2;
  Scheduler s(cfg, clock);
  s.register_worker(make_worker(WorkerId(1), WorkerBootId(1), 100));
  auto a1 = s.submit(make_spec(TenantId(1), ModelIdentity(10), RequestPhase::Prefill));
  auto a2 = s.submit(make_spec(TenantId(1), ModelIdentity(10), RequestPhase::Prefill));
  auto a3 = s.submit(make_spec(TenantId(1), ModelIdentity(10), RequestPhase::Prefill));
  REQUIRE(a1.ok() && a1.value().decision == AdmissionDecision::Admit, "a1 admit");
  REQUIRE(a2.ok() && a2.value().decision == AdmissionDecision::Admit, "a2 admit");
  REQUIRE(a3.ok() && a3.value().decision == AdmissionDecision::Defer, "a3 defer by global cap");
  REQUIRE(a3.ok() && a3.value().reason_code == "global_admission_saturated", "defer reason code");

  // deadline infeasible -> reject
  auto cfg2 = default_scheduler_config();
  Scheduler s2(cfg2, std::make_shared<SimulatedClock>());
  s2.register_worker(make_worker(WorkerId(2), WorkerBootId(2), 100));
  auto sp = make_spec(TenantId(3), ModelIdentity(20), RequestPhase::Prefill);
  sp.deadline = Deadline::after(std::chrono::milliseconds(10), clock->now());
  sp.demand.cost_units = 5000.0;
  auto b = s2.submit(sp);
  REQUIRE(b.ok() && b.value().decision == AdmissionDecision::Reject, "infeasible deadline reject");
  REQUIRE(b.ok() && b.value().reason_code == "deadline_infeasible", "infeasible reason");

  // token budget
  auto cfg3 = default_scheduler_config();
  cfg3.admission.max_global_token_budget = 100;
  Scheduler s3(cfg3, std::make_shared<SimulatedClock>());
  s3.register_worker(make_worker(WorkerId(3), WorkerBootId(3), 100));
  auto tspec = make_spec(TenantId(4), ModelIdentity(30), RequestPhase::Prefill);
  tspec.tokens.output = 200;
  auto t = s3.submit(tspec);
  REQUIRE(t.ok() && t.value().decision == AdmissionDecision::Defer, "token budget defer");
  REQUIRE(t.ok() && t.value().reason_code == "token_budget_exhausted", "token reason");

  // invalid spec
  auto bad = make_spec(TenantId(5), ModelIdentity(40), RequestPhase::Prefill);
  bad.tokens.input = -1;
  auto e = s3.submit(bad);
  REQUIRE(!e.ok(), "negative token rejected");

  return test_result("admission_test");
}
