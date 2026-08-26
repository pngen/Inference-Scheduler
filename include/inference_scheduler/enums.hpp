#pragma once

#include "error.hpp"
#include "result.hpp"

#include <string>
#include <string_view>

namespace inference_scheduler {

// ----------------------------------------------------------------------------
// Scheduling enums. Underlying values are stable integers for serialization.
// ----------------------------------------------------------------------------

enum class PriorityClass : int {
  Background = 0,
  Low = 1,
  Medium = 2,
  High = 3,
  Critical = 4,
};

enum class LatencyClass : int {
  Background = 0,
  Throughput = 1,
  Standard = 2,
  Interactive = 3,
};

// Which scheduling dimension dominates ordering for a given request.
enum class QueueClass : int {
  Latency = 0,
  Phase = 1,
  Tenant = 2,
  Priority = 3,
};

enum class RequestPhase : int {
  Unknown = 0,
  Prefill = 1,
  Decode = 2,
};

enum class RequestState : int {
  Created = 0,
  PendingAdmission = 1,
  Admitted = 2,
  Queued = 3,
  Reserved = 4,
  Dispatched = 5,
  Running = 6,
  Completing = 7,
  Completed = 8,
  Rejected = 9,
  Cancelled = 10,
  Expired = 11,
  Failed = 12,
};

enum class AttemptState : int {
  Created = 0,
  Reserved = 1,
  Dispatched = 2,
  Running = 3,
  Completing = 4,
  Succeeded = 5,
  Failed = 6,
  Cancelled = 7,
  Stale = 8,
  Expired = 9,
};

enum class WorkerState : int {
  Unknown = 0,
  Booted = 1,
  Online = 2,
  Busy = 3,
  Offline = 4,
  Drained = 5,
  Unhealthy = 6,
};

enum class AdmissionDecision : int {
  Admit = 0,
  Defer = 1,
  Reject = 2,
};

enum class DispatchDecision : int {
  Dispatch = 0,
  Hold = 1,
  NoEligibleWorker = 2,
  Cancelled = 3,
};

enum class RetryDecision : int {
  Retry = 0,
  GiveUp = 1,
  NoBudget = 2,
};

enum class CancellationReason : int {
  ClientRequested = 0,
  DeadlineExpired = 1,
  SchedulerDrain = 2,
  TransientFailure = 3,
  Policy = 4,
  Overload = 5,
  Manual = 6,
};

enum class CompletionStatus : int {
  Succeeded = 0,
  Failed = 1,
  Cancelled = 2,
  Expired = 3,
  Stale = 4,
  Ambiguous = 5,
};

enum class FailureClass : int {
  RetryableTransport = 0,
  RetryableWorker = 1,
  NonRetryableRequest = 2,
  SchedulerRejection = 3,
  DeadlineExpiration = 4,
  Cancellation = 5,
  StaleAuthority = 6,
  AmbiguousDispatch = 7,
  Internal = 8,
};

// ----------------------------------------------------------------------------
// name_of / from_string helpers.
// ----------------------------------------------------------------------------
constexpr std::int32_t to_int(PriorityClass e) noexcept { return static_cast<std::int32_t>(e); }
constexpr std::int32_t to_int(LatencyClass e) noexcept { return static_cast<std::int32_t>(e); }
constexpr std::int32_t to_int(RequestPhase e) noexcept { return static_cast<std::int32_t>(e); }
constexpr std::int32_t to_int(RequestState e) noexcept { return static_cast<std::int32_t>(e); }
constexpr std::int32_t to_int(AttemptState e) noexcept { return static_cast<std::int32_t>(e); }
constexpr std::int32_t to_int(WorkerState e) noexcept { return static_cast<std::int32_t>(e); }

const char* name_of(PriorityClass e) noexcept;
const char* name_of(LatencyClass e) noexcept;
const char* name_of(QueueClass e) noexcept;
const char* name_of(RequestPhase e) noexcept;
const char* name_of(RequestState e) noexcept;
const char* name_of(AttemptState e) noexcept;
const char* name_of(WorkerState e) noexcept;
const char* name_of(AdmissionDecision e) noexcept;
const char* name_of(DispatchDecision e) noexcept;
const char* name_of(RetryDecision e) noexcept;
const char* name_of(CancellationReason e) noexcept;
const char* name_of(CompletionStatus e) noexcept;
const char* name_of(FailureClass e) noexcept;

PriorityClass priority_class_from_string(std::string_view s) noexcept;
LatencyClass latency_class_from_string(std::string_view s) noexcept;
RequestPhase request_phase_from_string(std::string_view s) noexcept;

// True when a request/attempt state is terminal and can never become runnable.
constexpr bool is_terminal(RequestState s) noexcept {
  return s == RequestState::Completed || s == RequestState::Rejected ||
         s == RequestState::Cancelled || s == RequestState::Expired ||
         s == RequestState::Failed;
}
constexpr bool is_terminal(AttemptState s) noexcept {
  return s == AttemptState::Succeeded || s == AttemptState::Failed ||
         s == AttemptState::Cancelled || s == AttemptState::Stale ||
         s == AttemptState::Expired;
}

// A request is "active" (occupies capacity and may be scheduled) after it is
// admitted and before it reaches a terminal state.
constexpr bool is_active(RequestState s) noexcept {
  return s == RequestState::Admitted || s == RequestState::Queued ||
         s == RequestState::Reserved || s == RequestState::Dispatched ||
         s == RequestState::Running || s == RequestState::Completing;
}

}  // namespace inference_scheduler
