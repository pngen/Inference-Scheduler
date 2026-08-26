#pragma once

#include <cstdint>
#include <string>
#include <string_view>

namespace inference_scheduler {

// Stable error categories. Codes are intended to be serialized as their
// underlying integer so they survive persistence and process boundaries.
enum class ErrorCode : int {
  None = 0,
  InvalidArgument = 1,
  InvalidState = 2,
  NotFound = 3,
  AlreadyExists = 4,
  CapacityExhausted = 5,
  AdmissionRejected = 6,
  Deferred = 7,
  DeadlineExpired = 8,
  Cancelled = 9,
  StaleEpoch = 10,
  StaleWorker = 11,
  StaleAttempt = 12,
  Incompatible = 13,
  TransportFailure = 14,
  PersistenceFailure = 15,
  Corruption = 16,
  Unsupported = 17,
  Internal = 18,
};

constexpr std::int32_t to_int(ErrorCode c) noexcept { return static_cast<std::int32_t>(c); }

const char* error_code_name(ErrorCode c) noexcept;
ErrorCode error_code_from_name(std::string_view name) noexcept;
ErrorCode error_code_from_int(std::int32_t v) noexcept;

struct Error {
  Error() = default;
  Error(ErrorCode code, std::string message) noexcept
      : code(code), message(std::move(message)) {}
  Error(ErrorCode code, std::string message, std::string context) noexcept
      : code(code), message(std::move(message)), context(std::move(context)) {}

  ErrorCode code = ErrorCode::None;
  std::string message;
  std::string context;

  bool ok() const noexcept { return code == ErrorCode::None; }
  explicit operator bool() const noexcept { return !ok(); }
};

inline Error make_error(ErrorCode code, std::string message) {
  return Error(code, std::move(message));
}
inline Error make_error(ErrorCode code, std::string message, std::string context) {
  return Error(code, std::move(message), std::move(context));
}

}  // namespace inference_scheduler
