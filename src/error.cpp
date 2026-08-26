#include "inference_scheduler/error.hpp"

#include <cstring>
#include <string>

namespace inference_scheduler {

const char* error_code_name(ErrorCode c) noexcept {
  switch (c) {
    case ErrorCode::None: return "none";
    case ErrorCode::InvalidArgument: return "invalid_argument";
    case ErrorCode::InvalidState: return "invalid_state";
    case ErrorCode::NotFound: return "not_found";
    case ErrorCode::AlreadyExists: return "already_exists";
    case ErrorCode::CapacityExhausted: return "capacity_exhausted";
    case ErrorCode::AdmissionRejected: return "admission_rejected";
    case ErrorCode::Deferred: return "deferred";
    case ErrorCode::DeadlineExpired: return "deadline_expired";
    case ErrorCode::Cancelled: return "cancelled";
    case ErrorCode::StaleEpoch: return "stale_epoch";
    case ErrorCode::StaleWorker: return "stale_worker";
    case ErrorCode::StaleAttempt: return "stale_attempt";
    case ErrorCode::Incompatible: return "incompatible";
    case ErrorCode::TransportFailure: return "transport_failure";
    case ErrorCode::PersistenceFailure: return "persistence_failure";
    case ErrorCode::Corruption: return "corruption";
    case ErrorCode::Unsupported: return "unsupported";
    case ErrorCode::Internal: return "internal";
  }
  return "unknown";
}

ErrorCode error_code_from_name(std::string_view name) noexcept {
  if (name == "none") return ErrorCode::None;
  if (name == "invalid_argument") return ErrorCode::InvalidArgument;
  if (name == "invalid_state") return ErrorCode::InvalidState;
  if (name == "not_found") return ErrorCode::NotFound;
  if (name == "already_exists") return ErrorCode::AlreadyExists;
  if (name == "capacity_exhausted") return ErrorCode::CapacityExhausted;
  if (name == "admission_rejected") return ErrorCode::AdmissionRejected;
  if (name == "deferred") return ErrorCode::Deferred;
  if (name == "deadline_expired") return ErrorCode::DeadlineExpired;
  if (name == "cancelled") return ErrorCode::Cancelled;
  if (name == "stale_epoch") return ErrorCode::StaleEpoch;
  if (name == "stale_worker") return ErrorCode::StaleWorker;
  if (name == "stale_attempt") return ErrorCode::StaleAttempt;
  if (name == "incompatible") return ErrorCode::Incompatible;
  if (name == "transport_failure") return ErrorCode::TransportFailure;
  if (name == "persistence_failure") return ErrorCode::PersistenceFailure;
  if (name == "corruption") return ErrorCode::Corruption;
  if (name == "unsupported") return ErrorCode::Unsupported;
  if (name == "internal") return ErrorCode::Internal;
  return ErrorCode::Internal;
}

ErrorCode error_code_from_int(std::int32_t v) noexcept {
  if (v >= static_cast<std::int32_t>(ErrorCode::None) &&
      v <= static_cast<std::int32_t>(ErrorCode::Internal)) {
    return static_cast<ErrorCode>(v);
  }
  return ErrorCode::Internal;
}

}  // namespace inference_scheduler
