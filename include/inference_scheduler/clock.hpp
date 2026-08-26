#pragma once

#include <chrono>

namespace inference_scheduler {

// Monotonic time used for all elapsed-time scheduling semantics. A Clock is
// injectable so unit tests can drive a deterministic simulated clock while
// production code uses the system steady clock. Deadline policy never depends
// on the wall-clock time of day.
using ClockTime = std::chrono::steady_clock::time_point;
using ClockDuration = std::chrono::steady_clock::duration;

class Clock {
 public:
  virtual ~Clock() = default;
  virtual ClockTime now() const noexcept = 0;
};

class SystemClock final : public Clock {
 public:
  ClockTime now() const noexcept override { return std::chrono::steady_clock::now(); }
};

class SimulatedClock final : public Clock {
 public:
  explicit SimulatedClock(ClockTime start = ClockTime{}) : now_(start) {}

  ClockTime now() const noexcept override { return now_; }

  void advance(ClockDuration d) noexcept { now_ += d; }
  void set(ClockTime t) noexcept { now_ = t; }

 private:
  ClockTime now_;
};

}  // namespace inference_scheduler
