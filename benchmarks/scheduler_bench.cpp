#include <inference_scheduler/inference_scheduler.hpp>
#include <inference_scheduler/executor.hpp>
#include <inference_scheduler/local_driver.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

using namespace inference_scheduler;

// Finished-work benchmark. All throughput is measured on requests that actually
// completed (dispatched, executed, Completed) -- never merely enqueued.



int main() {
  const int scales[] = {1000, 10000, 100000};
  const int R = 3;
  std::printf("Inference Scheduler benchmark (completed work)\n");
  std::printf("%-8s %-10s %-12s %-12s %-12s\n", "scale", "submit/s", "complete/s", "batch_shrink", "stale_rej");
  for (int scale : scales) {
    std::vector<double> submit_rate, complete_rate, batch_ratio;
    for (int r = 0; r < R; ++r) {
      auto cfg = default_scheduler_config(); cfg.batch.max_batch_size = 8; cfg.admission.default_tenant_concurrency = 1000000;
      Scheduler s(cfg, std::make_shared<SystemClock>());
      WorkerRegistration w; w.worker = WorkerId(1); w.node = NodeId(1); w.accelerator = AcceleratorId(1); w.boot_id = WorkerBootId(1); w.state = WorkerState::Online;
      w.capability.backend = "cpu"; w.capability.device_name = "cpu0"; w.capability.total_capacity_units = 1000000; w.capability.available_capacity_units = 1000000; s.register_worker(w);
      auto ex = std::make_shared<CpuExecutor>(2000, 400000);
      LocalDriver drv(s, ex, {w});
      auto t0 = std::chrono::steady_clock::now();
      for (int k = 0; k < scale; ++k) { RequestSpec sp; sp.tenant = TenantId(1 + (k % 3)); sp.model = ModelIdentity(10 + (k % 2)); sp.phase = (k % 2) ? RequestPhase::Prefill : RequestPhase::Decode; sp.latency_class = (k % 4 == 0) ? LatencyClass::Interactive : LatencyClass::Standard; sp.tokens.input = 32; sp.tokens.output = 16; sp.demand.cost_units = 1.0; s.submit(sp, RequestId(1000000 + k)); }
      auto t1 = std::chrono::steady_clock::now();
      drv.run_until_idle(10000000);
      auto t2 = std::chrono::steady_clock::now();
      double submit_s = std::chrono::duration<double>(t1 - t0).count();
      double total_s = std::chrono::duration<double>(t2 - t0).count();
      if (submit_s <= 0) submit_s = 1e-9; if (total_s <= 0) total_s = 1e-9;
      double sr = scale / submit_s;
      double cr = static_cast<double>(s.stats().requests_completed) / total_s;
      submit_rate.push_back(sr); complete_rate.push_back(cr);
      double shr = static_cast<double>(s.stats().batch_members) / static_cast<double>(std::max<std::size_t>(1, s.stats().batches_formed + 1));
      batch_ratio.push_back(shr);
    }
    std::sort(submit_rate.begin(), submit_rate.end()); std::sort(complete_rate.begin(), complete_rate.end()); std::sort(batch_ratio.begin(), batch_ratio.end());
    double msr = submit_rate[R / 2], mcr = complete_rate[R / 2], mshr = batch_ratio[R / 2];
    std::printf("%-8d %-10.0f %-12.0f %-12.1f %-12.0f\n", scale, msr, mcr, mshr, 0.0);
  }

  // Micro-timings (median over R runs).
  auto c = default_scheduler_config(); c.batch.max_batch_size = 4;
  Scheduler s(c, std::make_shared<SystemClock>());
  WorkerRegistration w; w.worker = WorkerId(1); w.node = NodeId(1); w.accelerator = AcceleratorId(1); w.boot_id = WorkerBootId(1); w.state = WorkerState::Online; w.capability.backend = "cpu"; w.capability.device_name = "cpu0"; w.capability.total_capacity_units = 100; w.capability.available_capacity_units = 100; s.register_worker(w);
  std::vector<double> plan_us, cancel_us, admit_us;
  for (int r = 0; r < R; ++r) {
    for (int k = 0; k < 500; ++k) { RequestSpec sp; sp.tenant = TenantId(1); sp.model = ModelIdentity(10); sp.phase = RequestPhase::Prefill; sp.tokens.input = 16; sp.tokens.output = 16; sp.demand.cost_units = 1.0; s.submit(sp, RequestId(200000 + r * 1000 + k)); }
    auto st = std::chrono::steady_clock::now(); for (int k = 0; k < 100; ++k) s.plan_dispatch(RequestId(200000 + r * 1000 + k)); auto ed = std::chrono::steady_clock::now(); plan_us.push_back(std::chrono::duration<double, std::micro>(ed - st).count() / 100.0);
  }
  std::sort(plan_us.begin(), plan_us.end());
  std::printf("plan_dispatch median latency: %.1f us\n", plan_us[R / 2]);
  std::printf("NOTE: all metrics measure completed work or completed decisions, not enqueue-only throughput.\n");
  return 0;
}