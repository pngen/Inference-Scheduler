#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <limits>
#include <ostream>
#include <random>
#include <string>
#include <type_traits>

namespace inference_scheduler {

// ----------------------------------------------------------------------------
// Strong identifier types.
//
// An Id<Tag> wraps an unsigned 64-bit value. IDs are comparable, hashable,
// serializable (as the 64-bit value), and safe to carry across process
// boundaries. Uniqueness within a process is guaranteed by an IdFactory that
// combines a per-factory random salt (high 32 bits) with a monotonic counter
// (low 32 bits), so IDs issued by different factories essentially never
// collide. The salt is configurable so tests can reproduce deterministic ID
// sequences.
// ----------------------------------------------------------------------------

template <typename Tag>
class Id {
 public:
  using ValueType = std::uint64_t;

  constexpr Id() noexcept : value_{invalid} {}
  constexpr explicit Id(std::uint64_t value) noexcept : value_{value} {}

  constexpr std::uint64_t value() const noexcept { return value_; }
  constexpr bool valid() const noexcept { return value_ != invalid; }

  // 0 is reserved as the invalid/idle identifier.
  static constexpr std::uint64_t invalid = std::numeric_limits<std::uint64_t>::max();

  constexpr bool operator==(const Id& other) const noexcept { return value_ == other.value_; }
  constexpr bool operator!=(const Id& other) const noexcept { return value_ != other.value_; }
  constexpr bool operator<(const Id& other) const noexcept { return value_ < other.value_; }
  constexpr bool operator<=(const Id& other) const noexcept { return value_ <= other.value_; }
  constexpr bool operator>(const Id& other) const noexcept { return value_ > other.value_; }
  constexpr bool operator>=(const Id& other) const noexcept { return value_ >= other.value_; }

  // Numeric-identity comparison used as a stable, deterministic tie-break so
  // that scheduling decisions never depend on hash iteration order.
  std::size_t numeric_order() const noexcept { return static_cast<std::size_t>(value_); }

  std::string str() const;

 private:
  std::uint64_t value_;
};

template <typename Tag>
std::string Id<Tag>::str() const {
  return std::to_string(value_);
}

template <typename Tag>
std::ostream& operator<<(std::ostream& os, const Id<Tag>& id) {
  os << id.value();
  return os;
}

// ----------------------------------------------------------------------------
// IdFactory: issues unique IDs. Thread-safe. Owned by the agent (scheduler
// process, worker process, or client) so there is no hidden global mutable
// state; determinism is preserved by seeding the factory.
// ----------------------------------------------------------------------------
class IdFactory {
 public:
  IdFactory() : IdFactory(random_salt()) {}
  explicit IdFactory(std::uint64_t salt) : salt_{salt} {}

  template <typename Tag>
  Id<Tag> next() {
    // Combine salt (high) + counter (low). Counter starts at 1 so 0 stays
    // reserved. Wrap on overflow keeps the value unique for realistic runs.
    const std::uint64_t c = counter_.fetch_add(1, std::memory_order_relaxed) + 1;
    const std::uint64_t v = (salt_ << 32) | c;
    return Id<Tag>(v);
  }

  std::uint64_t salt() const noexcept { return salt_; }
  std::uint64_t counter() const noexcept { return counter_.load(std::memory_order_relaxed); }

  static std::uint64_t random_salt() {
    std::random_device rd;
    // Use 32 bits of entropy. This is only a collision-avoidance salt, not a
    // security primitive.
    const std::uint64_t r =
        (static_cast<std::uint64_t>(rd()) << 16) ^ static_cast<std::uint64_t>(rd());
    return (r & 0xFFFFFFFFu) | 0x80000000u;
  }

 private:
  std::uint64_t salt_;
  std::atomic<std::uint64_t> counter_{0};
};

// ----------------------------------------------------------------------------
// Tag declarations and concrete identifier types.
// ----------------------------------------------------------------------------
struct RequestIdTag {};
struct TenantIdTag {};
struct SessionIdTag {};
struct SequenceIdTag {};
struct AttemptIdTag {};
struct BatchIdTag {};
struct WorkerIdTag {};
struct NodeIdTag {};
struct AcceleratorIdTag {};
struct ModelIdentityTag {};
struct ModelRevisionTag {};
struct AdapterIdentityTag {};
struct CompatibilityKeyTag {};
struct SchedulerEpochTag {};
struct WorkerBootIdTag {};
struct GenerationTag {};
struct FaultDomainTag {};

using RequestId = Id<RequestIdTag>;
using TenantId = Id<TenantIdTag>;
using SessionId = Id<SessionIdTag>;
using SequenceId = Id<SequenceIdTag>;
using AttemptId = Id<AttemptIdTag>;
using BatchId = Id<BatchIdTag>;
using WorkerId = Id<WorkerIdTag>;
using NodeId = Id<NodeIdTag>;
using AcceleratorId = Id<AcceleratorIdTag>;
using ModelIdentity = Id<ModelIdentityTag>;
using ModelRevision = Id<ModelRevisionTag>;
using AdapterIdentity = Id<AdapterIdentityTag>;
using CompatibilityKey = Id<CompatibilityKeyTag>;
using SchedulerEpoch = Id<SchedulerEpochTag>;
using WorkerBootId = Id<WorkerBootIdTag>;
using Generation = Id<GenerationTag>;
using FaultDomain = Id<FaultDomainTag>;
using Pid = Id<FaultDomainTag>;

namespace detail {
constexpr std::uint64_t null_id_value() noexcept { return Id<RequestIdTag>::invalid; }
}  // namespace detail

}  // namespace inference_scheduler

namespace std {
template <typename Tag>
struct hash<inference_scheduler::Id<Tag>> {
  std::size_t operator()(const inference_scheduler::Id<Tag>& id) const noexcept {
    return std::hash<std::uint64_t>{}(id.value());
  }
};
}  // namespace std
