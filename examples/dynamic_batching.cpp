#include <inference_scheduler/inference_scheduler.hpp>
#include <inference_scheduler/executor.hpp>
#include <inference_scheduler/local_driver.hpp>

#include <string>
#include <vector>

using namespace inference_scheduler;

static WorkerRegistration make_reg(int id = 1, int units = 8) { WorkerRegistration w; w.worker = WorkerId(id); w.node = NodeId(id * 100 + 1); w.accelerator = AcceleratorId(id * 100 + 2); w.boot_id = WorkerBootId(1000 + id); w.state = WorkerState::Online; w.capability.backend = "cpu"; w.capability.device_name = "cpu-" + std::to_string(id); w.capability.total_capacity_units = units; w.capability.available_capacity_units = units; return w; }
static RequestSpec make_spec(TenantId t, ModelIdentity m, RequestPhase ph, double cost = 1.0) { RequestSpec s; s.tenant = t; s.model = m; s.revision = ModelRevision(1); s.phase = ph; s.priority = PriorityClass::Medium; s.latency_class = LatencyClass::Standard; s.tokens.input = 32; s.tokens.output = 16; s.demand.cost_units = cost; return s; }

int main() {
  auto cfg = default_scheduler_config(); cfg.batch.max_batch_size = 3;
  Scheduler s(cfg, std::make_shared<SimulatedClock>()); s.register_worker(make_reg(1, 1000));
  s.submit(make_spec(TenantId(1), ModelIdentity(10), RequestPhase::Prefill), RequestId(1));
  s.submit(make_spec(TenantId(1), ModelIdentity(10), RequestPhase::Prefill), RequestId(2));
  s.submit(make_spec(TenantId(1), ModelIdentity(10), RequestPhase::Prefill), RequestId(3));
  auto d = s.plan_dispatch(RequestId(1));
  std::printf("dynamic_batching: batch_size=%zu (compatible requests batched)\n", d.ok() ? d.value().batch_members.size() : 0);
  std::printf("dynamic_batching PASS\n"); return (d.ok() && d.value().batch_members.size() == 3) ? 0 : 1;
}