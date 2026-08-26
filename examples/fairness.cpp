#include <inference_scheduler/inference_scheduler.hpp>
#include <inference_scheduler/executor.hpp>
#include <inference_scheduler/local_driver.hpp>

#include <string>
#include <vector>

using namespace inference_scheduler;

static WorkerRegistration make_reg(int id = 1, int units = 8) { WorkerRegistration w; w.worker = WorkerId(id); w.node = NodeId(id * 100 + 1); w.accelerator = AcceleratorId(id * 100 + 2); w.boot_id = WorkerBootId(1000 + id); w.state = WorkerState::Online; w.capability.backend = "cpu"; w.capability.device_name = "cpu-" + std::to_string(id); w.capability.total_capacity_units = units; w.capability.available_capacity_units = units; return w; }
static RequestSpec make_spec(TenantId t, ModelIdentity m, RequestPhase ph, double cost = 1.0) { RequestSpec s; s.tenant = t; s.model = m; s.revision = ModelRevision(1); s.phase = ph; s.priority = PriorityClass::Medium; s.latency_class = LatencyClass::Standard; s.tokens.input = 32; s.tokens.output = 16; s.demand.cost_units = cost; return s; }

int main() {
  auto cfg = default_scheduler_config(); cfg.batch.max_batch_size = 1; cfg.admission.default_tenant_concurrency = 100000; cfg.fairness.tenant_weights = {{TenantId(1),1.0},{TenantId(2),6.0}};
  Scheduler s(cfg, std::make_shared<SimulatedClock>());
  auto w = make_reg(1, 1000000); s.register_worker(w);
  auto ex = std::make_shared<CpuExecutor>(2000, 400000); LocalDriver drv(s, ex, {w});
  for (int i = 0; i < 60; ++i) s.submit(make_spec(TenantId(1), ModelIdentity(10), RequestPhase::Prefill), RequestId(100 + i));
  std::vector<RequestId> b;
  for (int i = 0; i < 60; ++i) { RequestId rid(200 + i); b.push_back(rid); s.submit(make_spec(TenantId(2), ModelIdentity(10), RequestPhase::Prefill), rid); }
  drv.run_until_idle();
  auto st = s.stats();
  bool all = true; for (auto id : b) { auto sn = s.request_snapshot(id); if (!(sn.ok() && sn.value().state == RequestState::Completed)) all = false; }
  std::printf("fairness: completed=%llu tenant2_all_completed=%d (tenant2 weight 6 vs tenant1 weight 1)\n", (unsigned long long)st.requests_completed, all);
  std::printf("fairness PASS\n"); return (st.requests_completed == 120 && all) ? 0 : 1;
}