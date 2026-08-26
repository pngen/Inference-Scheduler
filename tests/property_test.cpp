#include "test_util.hpp"
#include <random>
#include <vector>
int main() {
  using namespace ts;
  const std::uint64_t seed = 0x9E3779B97F4A7C15ULL;  // recorded seed
  std::mt19937_64 rng(seed);
  auto clock = std::make_shared<SimulatedClock>();
  auto cfg = default_scheduler_config();
  cfg.batch.max_batch_size = 4;
  cfg.fairness.tenant_weights = { {TenantId(1), 1.0}, {TenantId(2), 1.5}, {TenantId(3), 0.5} };
  Scheduler s(cfg, clock);
  auto w1 = make_worker(WorkerId(1), WorkerBootId(1), 64);
  auto w2 = make_worker(WorkerId(2), WorkerBootId(2), 64);
  s.register_worker(w1); s.register_worker(w2);
  auto ex = std::make_shared<CpuExecutor>(2000, 400000);
  LocalDriver drv(s, ex, {w1, w2});
  std::vector<RequestId> active;
  int next_id = 1;
  const int MAX = 5000;
  for (int step = 0; step < MAX; ++step) {
    const int op = static_cast<int>(rng() % 6);
    if (op <= 1) {
      const int tenant = 1 + static_cast<int>(rng() % 3);
      const RequestPhase phase = (rng() % 2) ? RequestPhase::Prefill : RequestPhase::Decode;
      RequestId id(next_id++);
      auto spec = make_spec(TenantId(tenant), ModelIdentity(10 + (rng() % 3)), phase, 1.0 + (rng() % 4));
      if (rng() % 8 == 0) spec.deadline = Deadline::after(std::chrono::milliseconds(20 + rng() % 200), clock->now());
      auto ad = s.submit(spec, id);
      if (ad.ok() && ad.value().decision == AdmissionDecision::Admit) active.push_back(id);
    } else if (op == 2 && !active.empty()) {
      auto id = active[rng() % active.size()];
      s.cancel(id, CancellationReason::ClientRequested, "prop");
    } else if (op == 3) {
      drv.run_until_idle(1);
    } else if (op == 4) {
      clock->advance(std::chrono::milliseconds(3));
    } else {
      drv.run_until_idle(1);
    }
    auto st = s.stats();
    REQUIRE(st.current_queued >= 0 && st.current_admitted >= 0 && st.current_running >= 0 && st.current_reserved >= 0, "counters non-negative");
    REQUIRE(st.current_queued == static_cast<int>(s.queued_count()), "queued count consistent");
    REQUIRE(s.available_worker_capacity_units() >= 0, "capacity non-negative");
    auto snap = s.snapshot();
    for (const auto& ws : snap.workers) {
      REQUIRE(ws.reserved_units >= 0, "worker reserved non-negative");
      REQUIRE(ws.available_capacity_units - ws.reserved_units >= 0, "no oversubscription");
    }
  }
  // Drain; verify accounting closes to zero.
  drv.run_until_idle(100000);
  auto st = s.stats();
  REQUIRE(st.current_queued == 0, "queue drained");
  REQUIRE(st.current_admitted == 0, "no admitted leak");
  REQUIRE(st.current_running == 0 && st.current_reserved == 0, "no capacity leak");
  REQUIRE(s.available_worker_capacity_units() >= 0, "final capacity non-negative");
  return test_result("property_test");
}
