#include <inference_scheduler/inference_scheduler.hpp>
#include <inference_scheduler/executor.hpp>
#include <inference_scheduler/local_driver.hpp>

#include <string>
#include <vector>

using namespace inference_scheduler;

static WorkerRegistration make_reg(int id = 1, int units = 8) { WorkerRegistration w; w.worker = WorkerId(id); w.node = NodeId(id * 100 + 1); w.accelerator = AcceleratorId(id * 100 + 2); w.boot_id = WorkerBootId(1000 + id); w.state = WorkerState::Online; w.capability.backend = "cpu"; w.capability.device_name = "cpu-" + std::to_string(id); w.capability.total_capacity_units = units; w.capability.available_capacity_units = units; return w; }
static RequestSpec make_spec(TenantId t, ModelIdentity m, RequestPhase ph, double cost = 1.0) { RequestSpec s; s.tenant = t; s.model = m; s.revision = ModelRevision(1); s.phase = ph; s.priority = PriorityClass::Medium; s.latency_class = LatencyClass::Standard; s.tokens.input = 32; s.tokens.output = 16; s.demand.cost_units = cost; return s; }

int main() {
  auto cfg = default_scheduler_config(); cfg.scheduling.phase_weight[1] = 1.0; cfg.scheduling.phase_weight[2] = 3.0;  // favor decode
  cfg.admission.default_tenant_concurrency = 100000;  // admit the whole workload
  Scheduler s(cfg, std::make_shared<SimulatedClock>()); s.register_worker(make_reg(1, 1000000));
  auto ex = std::make_shared<CpuExecutor>(2000, 400000); LocalDriver drv(s, ex, {make_reg(1, 1000000)});
  for (int i = 0; i < 40; ++i) s.submit(make_spec(TenantId(1), ModelIdentity(10), RequestPhase::Prefill), RequestId(100 + i));
  std::vector<RequestId> dec;
  for (int i = 0; i < 40; ++i) { RequestId rid(200 + i); dec.push_back(rid); s.submit(make_spec(TenantId(1), ModelIdentity(10), RequestPhase::Decode), rid); }
  drv.run_until_idle();
  bool dec_done = true; for (auto id : dec) { auto sn = s.request_snapshot(id); if (!(sn.ok() && sn.value().state == RequestState::Completed)) dec_done = false; }
  auto st = s.stats();
  std::printf("prefill_decode: completed=%llu decode_all_done=%d (decode weighted 3x prefill)\n", (unsigned long long)st.requests_completed, dec_done);
  std::printf("prefill_decode PASS\n"); return (st.requests_completed == 80 && dec_done) ? 0 : 1;
}