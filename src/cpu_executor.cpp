#include "inference_scheduler/executor.hpp"

namespace inference_scheduler {

CpuExecutor::CpuExecutor(std::int64_t units_per_cost, std::int64_t max_units)
    : units_per_cost_(units_per_cost), max_units_(max_units) {}

std::string CpuExecutor::name() const { return "cpu"; }

std::vector<WorkResult> CpuExecutor::run(const std::vector<WorkRequest>& batch) {
  std::vector<WorkResult> out;
  out.reserve(batch.size());
  for (const auto& w : batch) {
    WorkResult r;
    r.request_id = w.request_id;
    r.attempt_id = w.attempt_id;
    r.generation = w.generation;
    r.output_tokens_produced = w.tokens.output;
    r.work_units = w.cost_units;
    std::int64_t work = w.cost_units * units_per_cost_;
    if (work > max_units_) work = max_units_;
    if (work < 0) work = 0;
    // Bounded, deterministic scalar computation so the host sees real CPU work.
    std::uint64_t acc = 0;
    for (std::int64_t i = 0; i < work; ++i) { acc += static_cast<std::uint64_t>(i & 0xFF); }
    r.duration_us = static_cast<std::int64_t>(acc & 0xFFFF);
    if (w.cancel_requested) {
      r.status = CompletionStatus::Cancelled;
      r.failure_class = FailureClass::Cancellation;
      r.error_message = "cancelled before execution";
    } else {
      r.status = CompletionStatus::Succeeded;
    }
    out.push_back(r);
  }
  return out;
}

}  // namespace inference_scheduler
