#include "inference_scheduler/enums.hpp"

#include <string>

namespace inference_scheduler {

const char* name_of(PriorityClass e) noexcept {
  switch (e) {
    case PriorityClass::Background: return "background";
    case PriorityClass::Low: return "low";
    case PriorityClass::Medium: return "medium";
    case PriorityClass::High: return "high";
    case PriorityClass::Critical: return "critical";
  }
  return "unknown";
}
const char* name_of(LatencyClass e) noexcept {
  switch (e) {
    case LatencyClass::Background: return "background";
    case LatencyClass::Throughput: return "throughput";
    case LatencyClass::Standard: return "standard";
    case LatencyClass::Interactive: return "interactive";
  }
  return "unknown";
}
const char* name_of(QueueClass e) noexcept {
  switch (e) {
    case QueueClass::Latency: return "latency";
    case QueueClass::Phase: return "phase";
    case QueueClass::Tenant: return "tenant";
    case QueueClass::Priority: return "priority";
  }
  return "unknown";
}
const char* name_of(RequestPhase e) noexcept {
  switch (e) {
    case RequestPhase::Unknown: return "unknown";
    case RequestPhase::Prefill: return "prefill";
    case RequestPhase::Decode: return "decode";
  }
  return "unknown";
}
const char* name_of(RequestState e) noexcept {
  switch (e) {
    case RequestState::Created: return "created";
    case RequestState::PendingAdmission: return "pending_admission";
    case RequestState::Admitted: return "admitted";
    case RequestState::Queued: return "queued";
    case RequestState::Reserved: return "reserved";
    case RequestState::Dispatched: return "dispatched";
    case RequestState::Running: return "running";
    case RequestState::Completing: return "completing";
    case RequestState::Completed: return "completed";
    case RequestState::Rejected: return "rejected";
    case RequestState::Cancelled: return "cancelled";
    case RequestState::Expired: return "expired";
    case RequestState::Failed: return "failed";
  }
  return "unknown";
}
const char* name_of(AttemptState e) noexcept {
  switch (e) {
    case AttemptState::Created: return "created";
    case AttemptState::Reserved: return "reserved";
    case AttemptState::Dispatched: return "dispatched";
    case AttemptState::Running: return "running";
    case AttemptState::Completing: return "completing";
    case AttemptState::Succeeded: return "succeeded";
    case AttemptState::Failed: return "failed";
    case AttemptState::Cancelled: return "cancelled";
    case AttemptState::Stale: return "stale";
    case AttemptState::Expired: return "expired";
  }
  return "unknown";
}
const char* name_of(WorkerState e) noexcept {
  switch (e) {
    case WorkerState::Unknown: return "unknown";
    case WorkerState::Booted: return "booted";
    case WorkerState::Online: return "online";
    case WorkerState::Busy: return "busy";
    case WorkerState::Offline: return "offline";
    case WorkerState::Drained: return "drained";
    case WorkerState::Unhealthy: return "unhealthy";
  }
  return "unknown";
}
const char* name_of(AdmissionDecision e) noexcept {
  switch (e) {
    case AdmissionDecision::Admit: return "admit";
    case AdmissionDecision::Defer: return "defer";
    case AdmissionDecision::Reject: return "reject";
  }
  return "unknown";
}
const char* name_of(DispatchDecision e) noexcept {
  switch (e) {
    case DispatchDecision::Dispatch: return "dispatch";
    case DispatchDecision::Hold: return "hold";
    case DispatchDecision::NoEligibleWorker: return "no_eligible_worker";
    case DispatchDecision::Cancelled: return "cancelled";
  }
  return "unknown";
}
const char* name_of(RetryDecision e) noexcept {
  switch (e) {
    case RetryDecision::Retry: return "retry";
    case RetryDecision::GiveUp: return "give_up";
    case RetryDecision::NoBudget: return "no_budget";
  }
  return "unknown";
}
const char* name_of(CancellationReason e) noexcept {
  switch (e) {
    case CancellationReason::ClientRequested: return "client_requested";
    case CancellationReason::DeadlineExpired: return "deadline_expired";
    case CancellationReason::SchedulerDrain: return "scheduler_drain";
    case CancellationReason::TransientFailure: return "transient_failure";
    case CancellationReason::Policy: return "policy";
    case CancellationReason::Overload: return "overload";
    case CancellationReason::Manual: return "manual";
  }
  return "unknown";
}
const char* name_of(CompletionStatus e) noexcept {
  switch (e) {
    case CompletionStatus::Succeeded: return "succeeded";
    case CompletionStatus::Failed: return "failed";
    case CompletionStatus::Cancelled: return "cancelled";
    case CompletionStatus::Expired: return "expired";
    case CompletionStatus::Stale: return "stale";
    case CompletionStatus::Ambiguous: return "ambiguous";
  }
  return "unknown";
}
const char* name_of(FailureClass e) noexcept {
  switch (e) {
    case FailureClass::RetryableTransport: return "retryable_transport";
    case FailureClass::RetryableWorker: return "retryable_worker";
    case FailureClass::NonRetryableRequest: return "non_retryable_request";
    case FailureClass::SchedulerRejection: return "scheduler_rejection";
    case FailureClass::DeadlineExpiration: return "deadline_expiration";
    case FailureClass::Cancellation: return "cancellation";
    case FailureClass::StaleAuthority: return "stale_authority";
    case FailureClass::AmbiguousDispatch: return "ambiguous_dispatch";
    case FailureClass::Internal: return "internal";
  }
  return "unknown";
}

PriorityClass priority_class_from_string(std::string_view s) noexcept {
  if (s == "background") return PriorityClass::Background;
  if (s == "low") return PriorityClass::Low;
  if (s == "medium") return PriorityClass::Medium;
  if (s == "high") return PriorityClass::High;
  if (s == "critical") return PriorityClass::Critical;
  return PriorityClass::Medium;
}
LatencyClass latency_class_from_string(std::string_view s) noexcept {
  if (s == "background") return LatencyClass::Background;
  if (s == "throughput") return LatencyClass::Throughput;
  if (s == "standard") return LatencyClass::Standard;
  if (s == "interactive") return LatencyClass::Interactive;
  return LatencyClass::Standard;
}
RequestPhase request_phase_from_string(std::string_view s) noexcept {
  if (s == "prefill") return RequestPhase::Prefill;
  if (s == "decode") return RequestPhase::Decode;
  return RequestPhase::Unknown;
}

}  // namespace inference_scheduler
