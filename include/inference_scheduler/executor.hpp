#pragma once

#include "clock.hpp"
#include "enums.hpp"
#include "id_types.hpp"
#include "types.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace inference_scheduler {

struct WorkRequest {
  RequestId request_id;
  AttemptId attempt_id;
  Generation generation;
  RequestPhase phase = RequestPhase::Prefill;
  ModelIdentity model;
  ModelRevision revision;
  AdapterIdentity adapter;
  TokenEstimate tokens;
  std::int64_t cost_units = 1;
  std::string payload;
  std::int32_t batch_index = 0;
  bool cancel_requested = false;
};

struct WorkResult {
  RequestId request_id;
  AttemptId attempt_id;
  Generation generation;
  CompletionStatus status = CompletionStatus::Succeeded;
  FailureClass failure_class = FailureClass::Internal;
  std::int64_t output_tokens_produced = 0;
  std::int64_t work_units = 0;
  std::int64_t duration_us = 0;
  std::string error_message;
};

class Executor {
 public:
  virtual ~Executor() = default;
  virtual std::string name() const = 0;
  virtual std::vector<WorkResult> run(const std::vector<WorkRequest>& batch) = 0;
};

// Deterministic CPU executor: real bounded scalar computation proportional to
// the cost estimate, used for tests, benchmarks, and as a reference backend.
class CpuExecutor final : public Executor {
 public:
  explicit CpuExecutor(std::int64_t units_per_cost = 2000, std::int64_t max_units = 400000);
  std::string name() const override;
  std::vector<WorkResult> run(const std::vector<WorkRequest>& batch) override;

 private:
  std::int64_t units_per_cost_;
  std::int64_t max_units_;
};

}  // namespace inference_scheduler
