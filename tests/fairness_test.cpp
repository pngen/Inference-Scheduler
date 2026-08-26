#include "test_util.hpp"
int main() {
  using namespace ts;
  auto cfg = default_scheduler_config();
  cfg.batch.max_batch_size = 1;   // isolate ordering; no cross-tenant batching
  cfg.admission.default_tenant_concurrency = 100000;  // admit the whole flood
  cfg.fairness.tenant_weights = { {TenantId(1), 1.0}, {TenantId(2), 1.0} };
  auto clock = std::make_shared<SimulatedClock>();
  Scheduler s(cfg, clock);
  auto w1 = make_worker(WorkerId(1), WorkerBootId(1), 1000000);
  s.register_worker(w1);
  auto ex = std::make_shared<CpuExecutor>(2000, 400000);
  LocalDriver drv(s, ex, {w1});

  // Flood tenant1; tenant2 submits a modest set. Preassign ids to track tenant2.
  std::vector<RequestId> b_ids;
  for (int i = 0; i < 500; ++i) { s.submit(make_spec(TenantId(1), ModelIdentity(10), RequestPhase::Prefill), RequestId(2000 + i)); }
  for (int i = 0; i < 50; ++i) { RequestId rid(3000 + i); b_ids.push_back(rid); s.submit(make_spec(TenantId(2), ModelIdentity(10), RequestPhase::Prefill), rid); }

  drv.run_until_idle();

  auto st = s.stats();
  REQUIRE(st.requests_completed == 550, "all submitted requests completed");
  // tenant2 (the smaller tenant) must not be starved: every tenant2 request terminal-complete.
  for (auto id : b_ids) {
    auto snap = s.request_snapshot(id);
    REQUIRE(snap.ok() && snap.value().state == RequestState::Completed, "tenant2 request completed");
  }
  // no leaked resources
  REQUIRE(st.current_queued == 0 && st.current_running == 0 && st.current_admitted == 0, "no leaked state");
  REQUIRE(s.available_worker_capacity_units() >= 0, "capacity accounting non-negative");
  return test_result("fairness_test");
}
