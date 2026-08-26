#pragma once

#include "clock.hpp"
#include "enums.hpp"
#include "id_types.hpp"

#include <array>
#include <chrono>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace inference_scheduler {

// ----------------------------------------------------------------------------
// Token and budget estimation.
// ----------------------------------------------------------------------------
struct TokenEstimate {
  std::int64_t input = 0;   // prefill tokens
  std::int64_t output = 0;  // decode tokens to produce
  std::int64_t decoded = 0; // decode tokens already produced

  std::int64_t total() const noexcept { return input + output; }
};

struct Budget {
  std::int64_t units = 0;  // abstract scheduling cost budget; <=0 means unbounded
  bool unbounded() const noexcept { return units <= 0; }
};

// ----------------------------------------------------------------------------
// Deadline: an absolute instant on the injectable clock. A request whose
// deadline is in the past must never be dispatched as if valid.
// ----------------------------------------------------------------------------
class Deadline {
 public:
  Deadline() = default;
  static Deadline none() { return Deadline{}; }
  static Deadline at(ClockTime p) {
    Deadline d;
    d.point_ = p;
    d.valid_ = true;
    return d;
  }
  static Deadline after(ClockDuration d, ClockTime now) { return Deadline::at(now + d); }

  bool valid() const noexcept { return valid_; }
  ClockTime point() const noexcept { return point_; }

  std::chrono::milliseconds remaining(ClockTime now) const noexcept {
    if (!valid_) return std::chrono::milliseconds::max();
    const auto d = point_ - now;
    if (d < std::chrono::steady_clock::duration::zero()) return std::chrono::milliseconds(0);
    return std::chrono::duration_cast<std::chrono::milliseconds>(d);
  }
  bool expired(ClockTime now) const noexcept {
    if (!valid_) return false;
    return now >= point_;
  }
  // Latency urgency; -1 when unknown. Lower is more urgent. Returns in ms.
  std::int64_t slack_ms(ClockTime now) const noexcept {
    if (!valid_) return -1;
    return std::chrono::duration_cast<std::chrono::milliseconds>(point_ - now).count();
  }

  friend bool operator==(const Deadline& a, const Deadline& b) noexcept {
    return a.valid_ == b.valid_ && (!a.valid_ || a.point_ == b.point_);
  }

 private:
  ClockTime point_{};
  bool valid_ = false;
};

// ----------------------------------------------------------------------------
// Locality and warmth hints supplied by callers.
// ----------------------------------------------------------------------------
struct LocalityHint {
  NodeId node;            // invalid() when unspecified
  AcceleratorId accelerator;  // invalid() when unspecified
  int affinity = 0;       // 0..100 preference strength

  bool has_preference() const noexcept {
    return node.valid() || accelerator.valid();
  }
};

struct WarmthHint {
  float model_warmth = 0.0f;  // 0..1 how likely model is resident/warm
  float state_warmth = 0.0f;  // 0..1 how likely KV/state is local
  float cache_hit = 0.0f;     // 0..1 expected state reuse
};

// ----------------------------------------------------------------------------
// Resource demand and snapshot.
// ----------------------------------------------------------------------------
struct ResourceDemand {
  TokenEstimate tokens;
  std::int64_t memory_bytes = 0;  // 0 when unknown, never trusted as exact
  double cost_units = 1.0;        // abstract execution cost, >= 0
};

struct ResourceSnapshot {
  int total_capacity_units = 0;      // 0 means "unknown / no limit"
  int available_capacity_units = 0;  // worker's own notion of free units
  std::int64_t memory_bytes_total = -1;   // -1 unknown
  std::int64_t memory_bytes_available = -1;
  int queue_depth = 0;
  int running = 0;
  bool healthy = true;
};

// ----------------------------------------------------------------------------
// Batching constraints.
// ----------------------------------------------------------------------------
struct BatchConstraints {
  int max_batch_size = 16;
  int max_tokens = 4096;                          // 0 = unlimited
  std::chrono::milliseconds max_wait{8};          // max batch-formation wait
  bool allow_phase_mix = false;                   // prefill+decode in one batch
  bool require_model_match = true;
  bool require_revision_match = true;
  bool require_adapter_match = true;
};

// ----------------------------------------------------------------------------
// Explicit, inspectable scheduling policy. Weights index the stable enum
// values so the policy is configurable and not hardcoded to class names.
// ----------------------------------------------------------------------------
inline std::size_t latency_index(LatencyClass c) noexcept {
  return static_cast<std::size_t>(static_cast<int>(c));
}
inline std::size_t priority_index(PriorityClass c) noexcept {
  return static_cast<std::size_t>(static_cast<int>(c));
}

// Explicit weights for accelerator-aware dispatch scoring. Each component is
// a named, inspectable factor; the ranker composes them, never a single opaque
// comparison.
struct DispatchWeights {
  double capability = 1.0;
  double capacity = 2.0;
  double load = 1.0;
  double deadline = 1.5;
  double latency_class = 1.0;
  double phase = 1.0;
  double locality = 0.5;
  double model_warmth = 0.6;
  double state_warmth = 0.6;
  double batch = 1.2;
  double fairness = 1.0;
};

struct SchedulingPolicy {
  // Weight per latency class (Background=0 .. Interactive=3). Higher weight =
  // more scheduling share for that class.
  std::array<double, 4> latency_weight{1.0, 1.0, 1.0, 1.0};
  // Weight per priority class (Background=0 .. Critical=4).
  std::array<double, 5> priority_weight{1.0, 1.0, 1.0, 1.0, 1.0};
  // Weight per request phase (Unknown=0, Prefill=1, Decode=2).
  std::array<double, 3> phase_weight{1.0, 1.0, 1.0};

  // Primary ordering dimension. Fairness by default; a coarse switch is not
  // exposed as one giant comparison because each dimension is its own weight
  // table composed by the ranker.
  QueueClass primary = QueueClass::Tenant;

  bool deadline_boost_enabled = true;
  double deadline_boost_max = 4.0;   // cap on deadline urgency multiplier
  std::int64_t deadline_boost_threshold_ms = 2000;  // slack below this gets boosted

  bool preemption_enabled = false;

  // Dispatch scoring weights.
  DispatchWeights dispatch;
};

struct TenantWeight {
  TenantId tenant;
  double weight = 1.0;
};

struct FairnessPolicy {
  std::vector<TenantWeight> tenant_weights;
  double default_weight = 1.0;
  int64_t quantum = 1;              // service units per round
  double starvation_guard_factor = 4.0;  // max virtual-runtime lead for a flow
  bool enabled = true;
};

struct RetryPolicy {
  int max_attempts = 3;
  std::chrono::milliseconds base_backoff{20};
  double backoff_multiplier = 2.0;
  std::chrono::milliseconds max_backoff{2000};
  bool retry_on_transport = true;
  bool retry_on_worker_failure = true;
  bool retry_on_ambiguous = false;
};

struct BackpressurePolicy {
  double admission_saturation_fraction = 0.95;  // admit only up to this fraction
  double queue_saturation_fraction = 1.0;
  bool emit_backpressure = true;
};

struct AdmissionLimits {
  int max_global_admitted = 1024;
  int max_global_queued = 4096;
  int max_tenants = 64;
  int default_tenant_concurrency = 64;
  int default_tenant_queue_depth = 512;
  std::int64_t max_global_token_budget = 0;  // 0 = unlimited
  bool enforce_deadline_feasibility = true;
};

struct SchedulerConfig {
  AdmissionLimits admission;
  SchedulingPolicy scheduling;
  FairnessPolicy fairness;
  BatchConstraints batch;
  RetryPolicy retry;
  BackpressurePolicy backpressure;
  bool persist_recovery = false;
  std::string state_path;  // used when persistence is enabled
  bool strict_deadline_rejection = true;  // reject infeasible rather than defer
};

// ----------------------------------------------------------------------------
// Request submission specification.
// ----------------------------------------------------------------------------
struct RequestSpec {
  TenantId tenant;
  SessionId session;
  SequenceId sequence;
  ModelIdentity model;
  ModelRevision revision;
  AdapterIdentity adapter;  // invalid() = none
  RequestPhase phase = RequestPhase::Prefill;
  PriorityClass priority = PriorityClass::Medium;
  LatencyClass latency_class = LatencyClass::Standard;
  TokenEstimate tokens;
  ResourceDemand demand;
  LocalityHint locality;
  WarmthHint warmth;
  Deadline deadline;
  Budget budget;
  std::int64_t estimated_memory_bytes = 0;
  std::string model_name;  // human-readable only
  std::string payload;     // opaque caller payload, never interpreted by core

  // Caller overrides to the scheduler's default batching for this request.
  int max_batch_size = 0;  // 0 = use scheduler default
  bool interactive = false;
};

// ----------------------------------------------------------------------------
// Admission output.
// ----------------------------------------------------------------------------
struct AdmissionOutput {
  AdmissionDecision decision = AdmissionDecision::Defer;
  std::string reason_code;          // stable, machine-readable code
  std::string explanation;          // human-readable explanation
  RequestId request_id;             // set when admitted
  AttemptId attempt_id;             // first attempt id when admitted
  Generation generation;            // first attempt generation
  std::chrono::milliseconds defer_after{0};
};

// ----------------------------------------------------------------------------
// Worker registration / capability advertisement.
// ----------------------------------------------------------------------------
struct WorkerCapability {
  AcceleratorId accelerator;
  std::string backend;           // "cpu" | "cuda"
  std::string device_name;
  int compute_capability_major = 0;
  int compute_capability_minor = 0;
  int total_capacity_units = 1;
  int available_capacity_units = 1;
  std::int64_t memory_bytes_total = -1;
  std::int64_t memory_bytes_available = -1;
  std::vector<ModelIdentity> models;      // empty = supports any
  std::vector<ModelRevision> revisions;   // empty = supports any
  std::vector<AdapterIdentity> adapters;  // empty = supports any / no adapter
  std::vector<RequestPhase> phases;       // empty = supports any
  BatchConstraints batch;
  std::string locality_tag;               // e.g. "node-a/gpu-0"
  float model_warmth = 0.0f;
  float state_warmth = 0.0f;
};

struct WorkerRegistration {
  WorkerId worker;
  NodeId node;
  AcceleratorId accelerator;
  WorkerBootId boot_id;
  WorkerCapability capability;
  WorkerState state = WorkerState::Booted;
  std::string endpoint;  // host:port when distributed
};

// ----------------------------------------------------------------------------
// Dispatch scoring and explanation.
// ----------------------------------------------------------------------------
struct ScoreComponent {
  std::string name;
  double value = 0.0;
  double weight = 0.0;
};

struct WorkerScore {
  WorkerId worker;
  bool eligible = false;
  std::vector<ScoreComponent> components;
  double total = 0.0;
  std::string reject_reason_code;
  std::string reject_reason;
};

struct ExplainResult {
  RequestId request_id;
  AttemptId attempt_id;
  AdmissionDecision admission = AdmissionDecision::Defer;
  bool admitted = false;
  std::string admission_reason_code;
  std::string admission_explanation;
  std::vector<std::string> ordering_factors;  // how it ranked in the queue
  std::vector<WorkerScore> candidates;
  WorkerId selected_worker;
  BatchId batch_id;
  bool batch_formed = false;
  std::string batch_explanation;
  SchedulerEpoch epoch;
  std::vector<std::string> notes;
};

struct DispatchOutcome {
  RequestId request_id;
  AttemptId attempt_id;
  Generation generation;
  WorkerId worker_id;
  BatchId batch_id;
  std::vector<RequestId> batch_members;  // when a batch was formed
  DispatchDecision decision = DispatchDecision::Hold;
  std::string reason_code;
  std::string explanation;
  std::vector<WorkerScore> candidates;  // populated when requested
  std::int64_t estimated_cost_units = 0;
};

// ----------------------------------------------------------------------------
// Events.
// ----------------------------------------------------------------------------
struct SchedulerEvent {
  std::uint64_t sequence = 0;
  SchedulerEpoch epoch;
  std::string type;      // machine-readable event type
  RequestId request_id;
  AttemptId attempt_id;
  WorkerId worker_id;
  BatchId batch_id;
  std::int64_t timestamp_ns = 0;  // monotonic
  std::string detail;
  std::string reason_code;
};

// ----------------------------------------------------------------------------
// Stats and snapshots.
// ----------------------------------------------------------------------------
struct LatencyClassStats {
  std::uint64_t admitted = 0;
  std::uint64_t completed = 0;
  std::uint64_t expired = 0;
};

struct SchedulerStats {
  std::uint64_t requests_submitted = 0;
  std::uint64_t requests_admitted = 0;
  std::uint64_t requests_deferred = 0;
  std::uint64_t requests_rejected = 0;
  std::uint64_t requests_completed = 0;
  std::uint64_t requests_cancelled = 0;
  std::uint64_t requests_expired = 0;
  std::uint64_t requests_failed = 0;
  std::uint64_t attempts_created = 0;
  std::uint64_t dispatches = 0;
  std::uint64_t completions = 0;
  std::uint64_t retries = 0;
  std::uint64_t stale_rejected = 0;
  std::uint64_t batches_formed = 0;
  std::uint64_t batch_members = 0;
  std::uint64_t backpressure_events = 0;

  int current_admitted = 0;
  int current_queued = 0;
  int current_reserved = 0;
  int current_running = 0;
  int current_capacity_units = 0;
  int current_reserved_units = 0;

  std::int64_t total_queue_delay_ms = 0;
  std::int64_t total_schedule_delay_ms = 0;
  std::int64_t total_dispatch_delay_ms = 0;

  std::array<LatencyClassStats, 4> by_latency{};
};

struct TenantSnapshot {
  TenantId tenant;
  double weight = 1.0;
  double virtual_time = 0.0;
  int queued = 0;
  int admitted = 0;
  std::int64_t services_consumed = 0;
  std::int64_t service_debt = 0;
};

struct WorkerSnapshot {
  WorkerId worker;
  NodeId node;
  WorkerBootId boot_id;
  WorkerState state;
  std::string backend;
  std::string device_name;
  int total_capacity_units = 0;
  int available_capacity_units = 0;
  int reserved_units = 0;
  int running = 0;
  std::int64_t memory_bytes_total = -1;
  std::int64_t memory_bytes_available = -1;
};

struct RequestSnapshot {
  RequestId request_id;
  TenantId tenant;
  RequestState state;
  RequestPhase phase;
  LatencyClass latency_class;
  PriorityClass priority;
  ModelIdentity model;
  ModelRevision revision;
  AttemptId attempt_id;
  Generation generation;
  WorkerId worker_id;
  Deadline deadline;
  std::int64_t queue_delay_ms = 0;
};

struct SchedulerSnapshot {
  SchedulerEpoch epoch;
  std::int64_t generated_ns = 0;
  SchedulerStats stats;
  std::vector<TenantSnapshot> tenants;
  std::vector<WorkerSnapshot> workers;
  std::vector<RequestSnapshot> requests;
};

}  // namespace inference_scheduler
