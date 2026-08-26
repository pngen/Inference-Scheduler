#pragma once

#include "clock.hpp"
#include "enums.hpp"
#include "error.hpp"
#include "id_types.hpp"
#include "result.hpp"
#include "types.hpp"

#include <chrono>
#include <functional>
#include <memory>
#include <optional>
#include <vector>

namespace inference_scheduler {

// ----------------------------------------------------------------------------
// Completion report: authoritative result from a worker for one attempt.
// Authority is bound to (epoch, worker boot id, request id, attempt id,
// generation); anything mismatched is rejected deterministically.
// ----------------------------------------------------------------------------
struct CompletionReport {
  SchedulerEpoch epoch;  // bound to the scheduler epoch that issued the attempt
  WorkerId worker;
  WorkerBootId boot_id;
  RequestId request_id;
  AttemptId attempt_id;
  Generation generation;
  CompletionStatus status = CompletionStatus::Succeeded;
  FailureClass failure_class = FailureClass::Internal;
  std::string error_message;
  std::int64_t output_tokens_produced = 0;
  std::int64_t work_units = 0;
  std::int64_t duration_us = 0;
  ResourceSnapshot worker_snapshot;  // worker's capacity after the attempt
};

enum class CompletionAcceptance {
  Accepted = 0,
  CancelledOutcome = 1,
  StaleEpoch = 2,
  StaleWorker = 3,
  StaleAttempt = 4,
  Duplicate = 5,
  Invalid = 6,
};

struct CompletionOutcome {
  CompletionAcceptance acceptance = CompletionAcceptance::Invalid;
  std::string reason_code;
  std::string explanation;
  bool retried = false;
  AttemptId retry_attempt;
};

// ----------------------------------------------------------------------------
// Scheduler: the authoritative scheduling runtime core. This class is
// thread-safe where documented. Callbacks (event handlers) are invoked
// outside any internal lock. All clock reads go through the injected Clock so
// unit tests can drive deterministic time.
// ----------------------------------------------------------------------------
class Scheduler {
 public:
  explicit Scheduler(SchedulerConfig config);
  Scheduler(SchedulerConfig config, std::shared_ptr<Clock> clock);
  Scheduler(SchedulerConfig config, std::shared_ptr<Clock> clock, std::optional<SchedulerEpoch> epoch);
  ~Scheduler();

  Scheduler(const Scheduler&) = delete;
  Scheduler& operator=(const Scheduler&) = delete;
  Scheduler(Scheduler&&) = delete;
  Scheduler& operator=(Scheduler&&) = delete;

  // --- configuration / identity -------------------------------------------------
  SchedulerConfig config() const;
  SchedulerEpoch epoch() const;
  std::shared_ptr<Clock> clock() const;

  // --- submission / admission ----------------------------------------------------
  // Submits a request; performs admission control. The returned AdmissionOutput
  // always carries an explicit decision (Admit / Defer / Reject) with reason
  // codes; the Result is an error only for invalid input or internal failure.
  Result<AdmissionOutput> submit(const RequestSpec& spec,
                                 std::optional<RequestId> preassigned = std::nullopt);

  // Applies admission to a request that was already created (exposed for
  // delegation and recovery). Advances Created -> Admitted -> Queued on admit.
  Result<AdmissionOutput> admit(const RequestSpec& spec,
                                std::optional<RequestId> preassigned = std::nullopt);

  // --- cancellation ------------------------------------------------------------
  Result<void> cancel(RequestId request, CancellationReason reason,
                      std::string detail = "");

  // --- workers ---------------------------------------------------------------
  void register_worker(const WorkerRegistration& registration);
  Result<void> update_worker_capability(WorkerId worker, const WorkerCapability& capability);
  Result<void> set_worker_state(WorkerId worker, WorkerState state,
                                const ResourceSnapshot* snapshot = nullptr);
  Result<void> unregister_worker(WorkerId worker);
  Result<WorkerSnapshot> worker_snapshot(WorkerId worker) const;

  // --- dispatch ---------------------------------------------------------------
  // Plans a dispatch for a runnable request: forms a compatible batch, ranks
  // eligible workers, and reserves capacity. On success the request(s) become
  // Reserved and a DispatchOutcome is returned. Available only for requests in
  // the runnable (Queued/Admitted) set; returns Hold for no eligible worker.
  Result<DispatchOutcome> plan_dispatch(RequestId request, bool include_explanation = false);
  Result<DispatchOutcome> plan_next_dispatch(bool include_explanation = false);

  // Confirms the worker accepted and started the work: Reserved -> Dispatched ->
  // Running. The worker must carry the exact attempt identity.
  Result<void> confirm_dispatch_start(const CompletionReport& start);

  // Processes an authoritative completion. Handles success, failure/retry,
  // cancellation races, and stale/duplicate rejection.
  Result<CompletionOutcome> complete_attempt(const CompletionReport& report);

  // --- inspection -------------------------------------------------------------
  Result<RequestSnapshot> request_snapshot(RequestId request) const;
  Result<RequestSpec> request_spec(RequestId request) const;
  SchedulerStats stats() const;
  SchedulerSnapshot snapshot() const;
  std::vector<SchedulerEvent> drain_events();  // non-blocking
  void set_event_handler(std::function<void(const SchedulerEvent&)> handler);
  std::size_t queued_count() const;      // runnable requests
  std::size_t reserved_count() const;
  int available_worker_capacity_units() const;

  // --- explain -----------------------------------------------------------------
  Result<ExplainResult> explain(RequestId request) const;

  // --- drain / shutdown / recovery ------------------------------------------
  // Marks the scheduler draining: no new work is admitted, running work may
  // complete, queued work is rejected according to policy.
  Result<void> begin_drain(std::string reason = "");
  Result<void> cancel_all_queued(std::string detail = "");
  // Deterministically brings recovered state up:
  Result<void> recover(const std::string& state_path);
  Result<void> persist(const std::string& state_path) const;  // fails when disabled

  // Simulate a scheduler epoch rollover; old worker boot identity authority
  // becomes stale.
  void roll_epoch();

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

// Build the default SchedulerConfig. A convenience for callers who want a
// sensible, explicit baseline rather than raw aggregate-initialization.
SchedulerConfig default_scheduler_config();

}  // namespace inference_scheduler
