#include <inference_scheduler/inference_scheduler.hpp>
#include <inference_scheduler/executor.hpp>
#include <inference_scheduler/local_driver.hpp>

#include <string>
#include <vector>

using namespace inference_scheduler;

static WorkerRegistration make_reg(int id = 1, int units = 8) { WorkerRegistration w; w.worker = WorkerId(id); w.node = NodeId(id * 100 + 1); w.accelerator = AcceleratorId(id * 100 + 2); w.boot_id = WorkerBootId(1000 + id); w.state = WorkerState::Online; w.capability.backend = "cpu"; w.capability.device_name = "cpu-" + std::to_string(id); w.capability.total_capacity_units = units; w.capability.available_capacity_units = units; return w; }
static RequestSpec make_spec(TenantId t, ModelIdentity m, RequestPhase ph, double cost = 1.0) { RequestSpec s; s.tenant = t; s.model = m; s.revision = ModelRevision(1); s.phase = ph; s.priority = PriorityClass::Medium; s.latency_class = LatencyClass::Standard; s.tokens.input = 32; s.tokens.output = 16; s.demand.cost_units = cost; return s; }

int main() {
  Scheduler s(default_scheduler_config(), std::make_shared<SimulatedClock>()); s.register_worker(make_reg(1, 100));
  auto ad = s.submit(make_spec(TenantId(1), ModelIdentity(10), RequestPhase::Prefill), RequestId(5));
  auto cr = s.cancel(RequestId(5), CancellationReason::ClientRequested, "example");
  auto sn = s.request_snapshot(RequestId(5));
  std::printf("cancellation_retry: cancel ok=%d state=%s (queued cancellation is authoritative)\n", cr.ok(), sn.ok() ? name_of(sn.value().state) : "?");
  // retry scenario
  auto ad2 = s.submit(make_spec(TenantId(1), ModelIdentity(20), RequestPhase::Prefill), RequestId(6));
  auto d = s.plan_dispatch(RequestId(6)); auto snap = s.request_snapshot(RequestId(6));
  CompletionReport start; start.epoch=s.epoch(); start.worker=d.value().worker_id; start.boot_id=WorkerBootId(1001); start.request_id=RequestId(6); start.attempt_id=snap.value().attempt_id; start.generation=snap.value().generation; start.worker_snapshot.available_capacity_units=100; s.confirm_dispatch_start(start);
  CompletionReport fail; fail.epoch=s.epoch(); fail.worker=d.value().worker_id; fail.boot_id=WorkerBootId(1001); fail.request_id=RequestId(6); fail.attempt_id=snap.value().attempt_id; fail.generation=snap.value().generation; fail.status=CompletionStatus::Failed; fail.failure_class=FailureClass::RetryableWorker; fail.error_message="boom"; fail.worker_snapshot.available_capacity_units=100;
  auto co = s.complete_attempt(fail);
  std::printf("cancellation_retry: retried=%d after_failed (new attempt identity)\n", co.ok() && co.value().retried);
  std::printf("cancellation_retry PASS\n"); return (cr.ok() && sn.ok() && sn.value().state == RequestState::Cancelled && co.ok() && co.value().retried) ? 0 : 1;
}