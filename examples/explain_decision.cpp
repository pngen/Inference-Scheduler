#include <inference_scheduler/inference_scheduler.hpp>
#include <inference_scheduler/executor.hpp>
#include <inference_scheduler/local_driver.hpp>

#include <string>
#include <vector>

using namespace inference_scheduler;

static WorkerRegistration make_reg(int id = 1, int units = 8) { WorkerRegistration w; w.worker = WorkerId(id); w.node = NodeId(id * 100 + 1); w.accelerator = AcceleratorId(id * 100 + 2); w.boot_id = WorkerBootId(1000 + id); w.state = WorkerState::Online; w.capability.backend = "cpu"; w.capability.device_name = "cpu-" + std::to_string(id); w.capability.total_capacity_units = units; w.capability.available_capacity_units = units; return w; }
static RequestSpec make_spec(TenantId t, ModelIdentity m, RequestPhase ph, double cost = 1.0) { RequestSpec s; s.tenant = t; s.model = m; s.revision = ModelRevision(1); s.phase = ph; s.priority = PriorityClass::Medium; s.latency_class = LatencyClass::Standard; s.tokens.input = 32; s.tokens.output = 16; s.demand.cost_units = cost; return s; }

int main() {
  Scheduler s(default_scheduler_config(), std::make_shared<SimulatedClock>());
  s.register_worker(make_reg(1, 8)); s.register_worker(make_reg(2, 8));
  auto ad = s.submit(make_spec(TenantId(1), ModelIdentity(10), RequestPhase::Prefill), RequestId(9));
  auto ex = s.explain(RequestId(9));
  std::printf("explain_decision: request=%llu admission=%s reason=%s\n", (unsigned long long)ex.value().request_id.value(), name_of(ex.value().admission), ex.value().admission_reason_code.c_str());
  for (const auto& c : ex.value().candidates) std::printf("  candidate worker=%llu eligible=%d score=%.3f reason=%s\n", (unsigned long long)c.worker.value(), c.eligible, c.total, c.reject_reason_code.c_str());
  std::printf("explain_decision PASS\n"); return ex.ok() ? 0 : 1;
}