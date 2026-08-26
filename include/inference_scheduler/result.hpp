#pragma once

#include "error.hpp"

#include <utility>

namespace inference_scheduler {

// ----------------------------------------------------------------------------
// Result<T>: an explicit error model. Normal control flow returns Result by
// value; failures are observable via ok()/error() rather than exceptions.
// ----------------------------------------------------------------------------
template <typename T>
class Result {
 public:
  Result() : value_{}, error_{ErrorCode::Internal, "uninitialized result"} {}
  Result(const T& value) : value_(value), error_{} {}
  Result(T&& value) : value_(std::move(value)), error_{} {}
  Result(const Error& err)
      : error_(err.code == ErrorCode::None ? Error{ErrorCode::Internal, err.message}
                                            : err) {}
  Result(Error&& err)
      : error_(err.code == ErrorCode::None ? Error{ErrorCode::Internal, err.message}
                                            : std::move(err)) {}

  static Result ok(T value) { return Result(std::move(value)); }
  static Result err(Error e) { return Result(std::move(e)); }

  bool ok() const noexcept { return error_.ok(); }
  bool has_value() const noexcept { return error_.ok(); }
  explicit operator bool() const noexcept { return error_.ok(); }

  T& value() { return value_; }
  const T& value() const { return value_; }

  Error error() const { return error_; }

 private:
  T value_{};
  Error error_;
};

// ----------------------------------------------------------------------------
// Result<void>: success carries no value.
// ----------------------------------------------------------------------------
template <>
class Result<void> {
 public:
  Result() noexcept : error_{} {}
  Result(const Error& err)
      : error_(err.code == ErrorCode::None ? Error{ErrorCode::Internal, err.message}
                                            : err) {}
  Result(Error&& err)
      : error_(err.code == ErrorCode::None ? Error{ErrorCode::Internal, err.message}
                                            : std::move(err)) {}

  static Result<void> success() { return Result<void>{}; }
  static Result<void> err(Error e) { return Result<void>(std::move(e)); }

  bool ok() const noexcept { return error_.ok(); }
  bool has_value() const noexcept { return error_.ok(); }
  void value() const noexcept {}
  explicit operator bool() const noexcept { return error_.ok(); }

  Error error() const { return error_; }

 private:
  Error error_;
};

}  // namespace inference_scheduler