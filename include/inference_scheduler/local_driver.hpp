#pragma once

#include "executor.hpp"
#include "scheduler.hpp"

#include <memory>
#include <vector>

namespace inference_scheduler {

// Wires a Scheduler to a local Executor so queued work runs in-process. This is
// the reference runtime used by examples and benchmarks; the distributed proof
// uses the same Scheduler over framed TCP instead.
class LocalDriver {
 public:
  LocalDriver(Scheduler& scheduler, std::shared_ptr<Executor> executor, std::vector<WorkerRegistration> workers);
  std::size_t run_until_idle(std::size_t max_batches = 1000000);
  void set_available_units(WorkerId worker, int capacity_units);
  std::size_t batches_run() const { return batches_run_; }
  std::size_t requests_completed() const { return requests_completed_; }

 private:
  const WorkerRegistration* find_worker(WorkerId id) const;
  bool dispatch_one(std::size_t& bcount);

  Scheduler& sched_;
  std::shared_ptr<Executor> ex_;
  std::vector<WorkerRegistration> workers_;
  std::vector<int> avail_units_;
  std::size_t batches_run_ = 0;
  std::size_t requests_completed_ = 0;
};

}  // namespace inference_scheduler
