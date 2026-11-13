#include "inference_scheduler/inference_scheduler.hpp"
#include "persistence.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <deque>
#include <functional>
#include <limits>
#include <map>
#include <mutex>
#include <queue>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace inference_scheduler {
namespace detail {

struct FlowKey {
  TenantId tenant;
  std::int32_t latency = 0;
  std::int32_t phase = 0;
  std::int32_t priority = 0;
  bool operator==(const FlowKey& o) const {
    return tenant == o.tenant && latency == o.latency && phase == o.phase && priority == o.priority;
  }
  bool operator<(const FlowKey& o) const {
    if (!(tenant == o.tenant)) return tenant < o.tenant;
    if (latency != o.latency) return latency < o.latency;
    if (phase != o.phase) return phase < o.phase;
    return priority < o.priority;
  }
};
struct FlowHasher {
  std::size_t operator()(const FlowKey& k) const {
    std::size_t h = std::hash<std::uint64_t>{}(k.tenant.value());
    h = h * 1000003u ^ static_cast<std::size_t>(k.latency);
    h = h * 1000003u ^ static_cast<std::size_t>(k.phase);
    h = h * 1000003u ^ static_cast<std::size_t>(k.priority);
    return h;
  }
};

struct ReadyEntry {
  RequestId id;
  Generation generation;
  std::int64_t slack;
  std::int32_t prio;
  std::uint64_t idv;
};
struct ReadyLess {
  bool operator()(const ReadyEntry& a, const ReadyEntry& b) const {
    if (a.slack != b.slack) return a.slack > b.slack;
    if (a.prio != b.prio) return a.prio < b.prio;
    return a.idv > b.idv;
  }
};

struct Flow {
  double weight = 1.0;
  double virtual_finish = 0.0;
  std::priority_queue<ReadyEntry, std::vector<ReadyEntry>, ReadyLess> pending;
};

struct CompatKey {
  std::uint64_t model = 0;
  std::uint64_t revision = 0;
  std::uint64_t adapter = 0;
  std::int32_t phase = 0;
  std::int32_t latency = 0;
  bool operator==(const CompatKey& o) const {
    return model == o.model && revision == o.revision && adapter == o.adapter && phase == o.phase && latency == o.latency;
  }
};
struct CompatHasher {
  std::size_t operator()(const CompatKey& k) const {
    std::size_t h = std::hash<std::uint64_t>{}(k.model);
    h = h * 1000003u ^ std::hash<std::uint64_t>{}(k.revision);
    h = h * 1000003u ^ std::hash<std::uint64_t>{}(k.adapter);
    h = h * 1000003u ^ static_cast<std::size_t>(k.phase);
    h = h * 1000003u ^ static_cast<std::size_t>(k.latency);
    return h;
  }
};

struct AttemptRecord {
  AttemptId attempt_id;
  Generation generation;
  AttemptState state = AttemptState::Created;
  WorkerId worker;
  BatchId batch;
  CompletionStatus final_status = CompletionStatus::Succeeded;
  FailureClass failure_class = FailureClass::Internal;
  std::string error_message;
  ClockTime created_at{};
  std::int64_t output_tokens = 0;
  std::int64_t work_units = 0;
  std::int64_t duration_us = 0;
};
struct RequestRecord {
  RequestSpec spec;
  RequestId request_id;
  RequestState state = RequestState::Created;
  AttemptId current_attempt;
  Generation generation;
  std::vector<AttemptRecord> history;
  ClockTime created_at{};
  ClockTime admitted_at{};
  ClockTime queued_at{};
  ClockTime reserved_at{};
  ClockTime started_at{};
  ClockTime completed_at{};
  std::int64_t queue_delay_ms = 0;
  std::int64_t schedule_delay_ms = 0;
  std::int64_t dispatch_delay_ms = 0;
  bool cancellation_requested = false;
  bool cancellation_committed = false;
  CancellationReason cancel_reason = CancellationReason::ClientRequested;
  std::string terminal_reason;
  bool in_batch = false;
  BatchId batch_id;
  int attempts_used = 0;
  std::uint64_t last_attempt_order = 0;
  bool counted_running = false;  // its tenant's running count includes it
};
struct TenantRecord {
  TenantId tenant;
  double weight = 1.0;
  int admitted = 0;
  int queued = 0;
  int running = 0;
  std::int64_t services_consumed = 0;
  std::int64_t service_debt = 0;
};
struct WorkerRecord {
  WorkerRegistration registration;
  WorkerState state = WorkerState::Unknown;
  ResourceSnapshot resource;
  std::int64_t reserved_units = 0;
  std::int64_t running_units = 0;
  std::map<BatchId, std::int64_t> batch_units;
  std::map<BatchId, int> batch_pending;
  std::map<BatchId, std::vector<RequestId>> batch_members;
  std::set<BatchId> started_batches;
  std::map<RequestId, AttemptId> in_flight;
};

}  // namespace detail

using detail::AttemptRecord;
using detail::RequestRecord;
using detail::TenantRecord;
using detail::WorkerRecord;
using detail::Flow;
using detail::FlowKey;
using detail::ReadyEntry;
using detail::ReadyLess;
using detail::FlowHasher;
using detail::CompatHasher;
using detail::CompatKey;

namespace {
struct BinWriter {
  std::string buf;
  void u8(std::uint8_t v) { buf.push_back(static_cast<char>(v)); }
  void u32(std::uint32_t v) { char b[4]; std::memcpy(b, &v, 4); buf.append(b, 4); }
  void u64(std::uint64_t v) { char b[8]; std::memcpy(b, &v, 8); buf.append(b, 8); }
  void i64(std::int64_t v) { char b[8]; std::memcpy(b, &v, 8); buf.append(b, 8); }
  void dbl(double v) { char b[8]; std::memcpy(b, &v, 8); buf.append(b, 8); }
  void i32(std::int32_t v) { u32(static_cast<std::uint32_t>(v)); }
  void yes(bool v) { u8(v ? 1 : 0); }
  void str(const std::string& s) { u64(static_cast<std::uint64_t>(s.size())); buf.append(s); }
  template <class IdT> void id(const IdT& v) { u64(v.value()); }
  template <class IdT> void opt_id(const IdT& v) { yes(v.valid()); if (v.valid()) id(v); }
};
struct BinReader {
  std::string_view s;
  std::size_t off = 0;
  bool ok = true;
  bool can(std::size_t n) const { return ok && off + n <= s.size(); }
  std::uint8_t u8() { if (!can(1)) { ok = false; return 0; } return static_cast<std::uint8_t>(s[off++]); }
  std::uint32_t u32() { if (!can(4)) { ok = false; return 0; } std::uint32_t v; std::memcpy(&v, s.data() + off, 4); off += 4; return v; }
  std::uint64_t u64() { if (!can(8)) { ok = false; return 0; } std::uint64_t v; std::memcpy(&v, s.data() + off, 8); off += 8; return v; }
  std::int64_t i64() { return static_cast<std::int64_t>(u64()); }
  double dbl() { if (!can(8)) { ok = false; return 0.0; } double v; std::memcpy(&v, s.data() + off, 8); off += 8; return v; }
  std::int32_t i32() { return static_cast<std::int32_t>(u32()); }
  bool yes() { return u8() != 0; }
  std::string str() { std::uint64_t n = u64(); if (!ok || n > 16u * 1024u * 1024u) { ok = false; return {}; } if (off + n > s.size()) { ok = false; return {}; } std::string r(s.data() + off, static_cast<std::size_t>(n)); off += static_cast<std::size_t>(n); return r; }
  template <class Tag> Id<Tag> id() { return Id<Tag>(u64()); }
  template <class Tag> void opt_id(Id<Tag>& out) { bool p = yes(); if (p) out = Id<Tag>(u64()); else out = Id<Tag>(); }
};
}  // namespace

namespace {
std::int64_t ms_between(ClockTime a, ClockTime b) {
  return std::chrono::duration_cast<std::chrono::milliseconds>(b - a).count();
}
double clamp01(double v) { return v < 0.0 ? 0.0 : (v > 1.0 ? 1.0 : v); }
std::int64_t ceil_cost(double c) { return c <= 0.0 ? 1 : static_cast<std::int64_t>(std::ceil(c)); }
int phase_int(RequestPhase p) { return static_cast<int>(p); }
int latency_int(LatencyClass c) { return static_cast<int>(c); }
int priority_int(PriorityClass c) { return static_cast<int>(c); }

// Whether an integer is one of an enum's values. Each of these enums counts up
// from zero, so its last enumerator bounds the range.
template <class E>
bool enum_in_range(std::int64_t v, E last) { return v >= 0 && v <= static_cast<std::int64_t>(last); }
bool class_values_ok(const RequestSpec& s) {
  return enum_in_range(phase_int(s.phase), RequestPhase::Decode) &&
         enum_in_range(priority_int(s.priority), PriorityClass::Critical) &&
         enum_in_range(latency_int(s.latency_class), LatencyClass::Interactive);
}
// Sum of two non-negative counts that stops at the top of the range.
std::int64_t sat_add(std::int64_t a, std::int64_t b) {
  const std::int64_t top = std::numeric_limits<std::int64_t>::max();
  return a > top - b ? top : a + b;
}
}  // namespace

class Scheduler::Impl {
 public:
  Impl(SchedulerConfig cfg, std::shared_ptr<Clock> clock, std::optional<SchedulerEpoch> epoch)
      : cfg_(std::move(cfg)),
        clock_(clock ? std::move(clock) : std::make_shared<SystemClock>()),
        id_factory_(),
        epoch_value_(1) {
    if (epoch && epoch->valid()) epoch_value_ = epoch->value();
    if (cfg_.state_path.empty()) cfg_.state_path = "inference_scheduler_state.bin";
  }
  ~Impl() = default;

  SchedulerConfig cfg_;
  std::shared_ptr<Clock> clock_;
  std::uint64_t epoch_value_ = 1;
  IdFactory id_factory_;
  double virtual_time_ = 0.0;

  mutable std::mutex mu_;
  std::unordered_map<RequestId, RequestRecord> requests_;
  std::unordered_map<WorkerId, WorkerRecord> workers_;
  std::unordered_map<TenantId, TenantRecord> tenants_;
  std::unordered_map<FlowKey, Flow, FlowHasher> flows_;
  std::unordered_map<CompatKey, std::unordered_set<RequestId>, CompatHasher> compat_;
  std::set<FlowKey> active_flows_;
  std::unordered_set<RequestId> queued_;

  std::int64_t global_admitted_ = 0;
  std::int64_t global_queued_ = 0;
  std::int64_t global_reserved_units_ = 0;
  std::int64_t global_running_units_ = 0;
  std::int64_t global_tokens_ = 0;
  std::int64_t active_tenants_ = 0;  // tenants holding at least one admitted request

  bool draining_ = false;
  std::string drain_reason_;
  int recovery_mode_ = 0;  // 0 normal, 1 recovering

  mutable std::mutex event_mu_;
  std::deque<SchedulerEvent> event_log_;
  std::function<void(const SchedulerEvent&)> event_handler_;
  bool event_stream_mode_ = false;
  std::uint64_t event_seq_ = 0;

  SchedulerStats stats_;

  double tenant_weight(TenantId t) const {
    for (const auto& c : cfg_.fairness.tenant_weights) { if (c.tenant == t) return c.weight; }
    return cfg_.fairness.default_weight;
  }
  double flow_weight(const RequestSpec& s) const {
    const auto& p = cfg_.scheduling;
    return tenant_weight(s.tenant) * p.latency_weight[latency_index(s.latency_class)] *
           p.priority_weight[priority_index(s.priority)] * p.phase_weight[phase_index(s.phase)];
  }
  static std::size_t phase_index(RequestPhase ph) { return static_cast<std::size_t>(phase_int(ph)); }
  FlowKey flow_key(const RequestSpec& s) const {
    FlowKey k; k.tenant = s.tenant; k.latency = latency_int(s.latency_class);
    k.phase = phase_int(s.phase); k.priority = priority_int(s.priority); return k;
  }
  CompatKey compat_key(const RequestSpec& s) const {
    CompatKey k;
    k.model = s.model.value(); k.revision = s.revision.value();
    k.adapter = s.adapter.valid() ? s.adapter.value() : 0;
    k.phase = cfg_.batch.allow_phase_mix ? 256 : phase_int(s.phase);
    k.latency = latency_int(s.latency_class);
    return k;
  }
  RequestRecord& rec(RequestId id) { return requests_[id]; }
  const RequestRecord& crec(RequestId id) const { return requests_.at(id); }

  // ----- admission -----
  Result<AdmissionOutput> do_submit(const RequestSpec& spec, std::optional<RequestId> pre) {
    std::lock_guard<std::mutex> g(mu_);
    auto check = validate_spec(spec);
    if (!check) return Result<AdmissionOutput>::err(check.error());
    if (pre) {
      if (!pre->valid()) return Result<AdmissionOutput>::err(make_error(ErrorCode::InvalidArgument, "preassigned request id is not valid"));
      // A request id names one request for good. Submitting it again must not
      // disturb the request that already holds it.
      if (requests_.count(*pre) != 0) return Result<AdmissionOutput>::err(make_error(ErrorCode::AlreadyExists, "request id already in use"));
    }
    AdmissionOutput out;
    RequestId rid = pre ? *pre : id_factory_.next<RequestIdTag>();
    const ClockTime now = clock_->now();
    auto ad = make_admission(spec, now);
    out.decision = ad.first; out.reason_code = ad.second;
    if (ad.first == AdmissionDecision::Admit) {
      RequestRecord rec;
      rec.spec = spec; rec.request_id = rid;
      rec.state = RequestState::Admitted; rec.generation = id_factory_.next<GenerationTag>();
      rec.created_at = now; rec.admitted_at = now;
      rec.queue_delay_ms = 0;
      stats_.requests_submitted += 1; stats_.requests_admitted += 1;
      stats_.by_latency[latency_index(spec.latency_class)].admitted += 1;
      if (spec.deadline.valid() && spec.deadline.expired(now)) {
        rec.state = RequestState::Expired; rec.terminal_reason = "deadline_expired_at_admit";
        requests_.insert({rid, std::move(rec)});
        stats_.requests_expired += 1;
        emit_event("expired", rid, AttemptId(), WorkerId(), BatchId(), rec.terminal_reason);
        out.decision = AdmissionDecision::Reject; out.reason_code = "deadline_expired_at_submit";
        out.explanation = "request deadline already expired at admission";
        return Result<AdmissionOutput>::ok(std::move(out));
      }
      rec.current_attempt = id_factory_.next<AttemptIdTag>();
      rec.state = RequestState::Queued; rec.queued_at = now;
      rec.attempts_used = 0;
      AttemptRecord a0; a0.attempt_id = rec.current_attempt; a0.generation = rec.generation;
      a0.state = AttemptState::Created; a0.created_at = now;
      rec.history.push_back(a0);
      stats_.attempts_created += 1;
      // push then enqueue (requests_ owns the record; enqueue mutates it)
      requests_.insert({rid, std::move(rec)});
      enqueue_request(requests_[rid]);
      hold_admission(spec);
      out.request_id = rid; out.attempt_id = requests_[rid].current_attempt; out.generation = requests_[rid].generation;
      out.explanation = "admitted; queued for scheduling";
      emit_event("admitted", rid, out.attempt_id, WorkerId(), BatchId(), "admitted");
      emit_event("queued", rid, out.attempt_id, WorkerId(), BatchId(), "queued");
    } else if (ad.first == AdmissionDecision::Defer) {
      stats_.requests_submitted += 1; stats_.requests_deferred += 1;
      out.defer_after = std::chrono::milliseconds(25);
      out.explanation = ad.second;
    } else {
      stats_.requests_submitted += 1; stats_.requests_rejected += 1;
      RequestRecord rec; rec.spec = spec; rec.request_id = rid;
      rec.state = RequestState::Rejected; rec.terminal_reason = ad.second;
      rec.created_at = now; rec.generation = id_factory_.next<GenerationTag>();
      requests_.insert({rid, std::move(rec)});
      out.explanation = ad.second;
      emit_event("rejected", rid, AttemptId(), WorkerId(), BatchId(), ad.second);
    }
    return Result<AdmissionOutput>::ok(std::move(out));
  }

  Result<void> validate_spec(const RequestSpec& s) const {
    if (!s.tenant.valid()) return Result<void>::err(make_error(ErrorCode::InvalidArgument, "tenant id required"));
    if (!s.model.valid()) return Result<void>::err(make_error(ErrorCode::InvalidArgument, "model identity required"));
    if (s.tokens.input < 0 || s.tokens.output < 0 || s.tokens.decoded < 0) return Result<void>::err(make_error(ErrorCode::InvalidArgument, "negative token estimate"));
    // Input and output are summed for the token budget and the batch token cap.
    if (s.tokens.input > std::numeric_limits<std::int64_t>::max() - s.tokens.output) return Result<void>::err(make_error(ErrorCode::InvalidArgument, "token estimate out of range"));
    if (s.demand.cost_units < 0.0) return Result<void>::err(make_error(ErrorCode::InvalidArgument, "negative cost units"));
    // Worker capacity is counted in int units: a cost beyond that range, or one
    // that is not a number, can never be placed on any worker.
    if (!(s.demand.cost_units <= static_cast<double>(std::numeric_limits<int>::max()))) return Result<void>::err(make_error(ErrorCode::InvalidArgument, "cost units out of range"));
    if (s.estimated_memory_bytes < 0) return Result<void>::err(make_error(ErrorCode::InvalidArgument, "negative memory estimate"));
    if (s.max_batch_size < 0) return Result<void>::err(make_error(ErrorCode::InvalidArgument, "negative max batch size"));
    // The class values index the policy weight tables and per-class statistics.
    if (!class_values_ok(s)) return Result<void>::err(make_error(ErrorCode::InvalidArgument, "phase, priority or latency class out of range"));
    return Result<void>::success();
  }

  // A request holds one admission slot, one of its tenant's, and its tokens of
  // the global budget from the moment it is admitted until it becomes terminal.
  void hold_admission(const RequestSpec& spec) {
    global_admitted_ += 1;
    TenantRecord& t = tenant_of(spec.tenant);
    if (t.admitted == 0) active_tenants_ += 1;
    t.admitted += 1;
    global_tokens_ = sat_add(global_tokens_, spec.tokens.total());
  }
  void release_admission(const RequestRecord& rec) {
    if (global_admitted_ > 0) global_admitted_ -= 1;
    TenantRecord& t = tenant_of(rec.spec.tenant);
    if (t.admitted > 0) { t.admitted -= 1; if (t.admitted == 0 && active_tenants_ > 0) active_tenants_ -= 1; }
    global_tokens_ -= std::min(global_tokens_, rec.spec.tokens.total());
  }
  // The tenant's running count covers a request from its confirmed start until
  // that attempt ends, however it ends.
  void leave_running(RequestRecord& rec) {
    if (!rec.counted_running) return;
    rec.counted_running = false;
    TenantRecord& t = tenant_of(rec.spec.tenant);
    if (t.running > 0) t.running -= 1;
  }

  std::pair<AdmissionDecision, std::string> make_admission(const RequestSpec& spec, ClockTime now) const {
    const AdmissionLimits& a = cfg_.admission;
    if (draining_) return {AdmissionDecision::Reject, "scheduler_draining"};
    if (recovery_mode_ == 1) return {AdmissionDecision::Reject, "recovering"};
    if (spec.deadline.valid() && spec.deadline.expired(now)) {
      return {AdmissionDecision::Reject, "deadline_expired_at_submit"};
    }
    if (cfg_.admission.enforce_deadline_feasibility && spec.deadline.valid()) {
      const auto rem = spec.deadline.remaining(now);
      const std::int64_t cost = ceil_cost(spec.demand.cost_units);
      if (rem.count() < 0) return {AdmissionDecision::Reject, "deadline_impossible"};
      if (rem.count() > 0 && cost > static_cast<std::int64_t>(rem.count())) {
        if (cfg_.strict_deadline_rejection) {
          return {AdmissionDecision::Reject, "deadline_infeasible"};
        }
        return {AdmissionDecision::Defer, "deadline_infeasible"};
      }
    }
    if (global_admitted_ >= a.max_global_admitted) {
      return {AdmissionDecision::Defer, "global_admission_saturated"};
    }
    if (global_queued_ >= a.max_global_queued) {
      return {AdmissionDecision::Defer, "global_queue_saturated"};
    }
    if (a.max_global_token_budget > 0 &&
        spec.tokens.total() > a.max_global_token_budget - std::min(global_tokens_, a.max_global_token_budget)) {
      return {AdmissionDecision::Defer, "token_budget_exhausted"};
    }
    TenantId t = spec.tenant;
    auto it = tenants_.find(t);
    // A tenant holding no admitted work would become one more active tenant.
    if (a.max_tenants > 0 && (it == tenants_.end() || it->second.admitted == 0) &&
        active_tenants_ >= a.max_tenants) {
      return {AdmissionDecision::Defer, "tenant_limit_reached"};
    }
    const int conc = a.default_tenant_concurrency;
    const int depth = a.default_tenant_queue_depth;
    if (it != tenants_.end() && it->second.admitted >= conc) {
      return {AdmissionDecision::Defer, "tenant_concurrency_saturated"};
    }
    if (it != tenants_.end() && it->second.queued >= depth) {
      return {AdmissionDecision::Defer, "tenant_queue_depth_saturated"};
    }
    // Capability / model compatibility (only when workers are registered).
    if (workers_.empty()) return {AdmissionDecision::Admit, "admitted"};
    bool any = false;
    for (const auto& [wk, wr] : workers_) { if (compatible(spec, wr.registration.capability)) any = true; }
    if (!any) return {AdmissionDecision::Defer, "no_compatible_worker"};
    return {AdmissionDecision::Admit, "admitted"};
  }

  bool compatible(const RequestSpec& spec, const WorkerCapability& cap) const {
    if (cap.total_capacity_units <= 0) return false;
    if (!cap.models.empty()) {
      bool have = false;
      for (const auto& m : cap.models) if (m == spec.model) { have = true; break; }
      if (!have) return false;
    }
    if (!cap.revisions.empty()) {
      bool have = false;
      for (const auto& m : cap.revisions) if (m == spec.revision) { have = true; break; }
      if (!have) return false;
    }
    if (spec.adapter.valid() && !cap.adapters.empty()) {
      bool have = false;
      for (const auto& m : cap.adapters) if (m == spec.adapter) { have = true; break; }
      if (!have) return false;
    }
    if (!cap.phases.empty()) {
      bool have = false;
      for (const auto& p : cap.phases) if (p == spec.phase) { have = true; break; }
      if (!have) return false;
    }
    return true;
  }

  // ----- queue / selection -----
  TenantRecord& tenant_of(TenantId t) {
    auto it = tenants_.find(t);
    if (it == tenants_.end()) { TenantRecord tr; tr.tenant = t; tr.weight = tenant_weight(t); it = tenants_.insert({t, tr}).first; }
    return it->second;
  }
  void enqueue_request(RequestRecord& rec) {
    const FlowKey k = flow_key(rec.spec);
    Flow& f = flows_[k];
    f.weight = flow_weight(rec.spec);
    const ClockTime now = clock_->now();
    ReadyEntry e;
    e.id = rec.request_id; e.generation = rec.generation;
    e.slack = rec.spec.deadline.valid() ? rec.spec.deadline.remaining(now).count() : std::numeric_limits<std::int64_t>::max();
    e.prio = priority_int(rec.spec.priority); e.idv = rec.request_id.value();
    if (f.pending.empty()) { f.virtual_finish = std::max(f.virtual_finish, virtual_time_); active_flows_.insert(k); }
    f.pending.push(e);
    queued_.insert(rec.request_id);
    compat_[compat_key(rec.spec)].insert(rec.request_id);
    rec.state = RequestState::Queued; rec.queued_at = clock_->now();
    global_queued_ += 1;
    tenant_of(rec.spec.tenant).queued += 1;
  }
  void dequeue_request(RequestRecord& rec) {
    if (queued_.erase(rec.request_id) > 0) global_queued_ -= 1;
    const CompatKey k = compat_key(rec.spec);
    auto it = compat_.find(k);
    if (it != compat_.end()) { it->second.erase(rec.request_id); if (it->second.empty()) compat_.erase(it); }
    auto t = tenants_.find(rec.spec.tenant);
    if (t != tenants_.end() && t->second.queued > 0) t->second.queued -= 1;
  }
  void expire_request(RequestRecord& rec) {
    if (rec.state != RequestState::Queued) return;
    dequeue_request(rec);
    rec.state = RequestState::Expired; rec.terminal_reason = "deadline_expired";
    if (!rec.history.empty()) rec.history.back().state = AttemptState::Expired;
    release_admission(rec);
    stats_.requests_expired += 1;
    stats_.by_latency[latency_index(rec.spec.latency_class)].expired += 1;
    if (cfg_.batch.max_wait.count() >= 0 && rec.spec.deadline.valid()) emit_event("expired", rec.request_id, rec.current_attempt, WorkerId(), BatchId(), "deadline_expired");
  }
  void expire_scan(ClockTime now) {
    std::vector<RequestId> to;
    for (auto id : queued_) {
      auto it = requests_.find(id);
      if (it != requests_.end() && it->second.state == RequestState::Queued && it->second.spec.deadline.valid() && it->second.spec.deadline.expired(now)) to.push_back(id);
    }
    for (auto id : to) { auto it = requests_.find(id); if (it != requests_.end()) expire_request(it->second); }
  }
  double effective_finish(const Flow& f, const RequestRecord& r, ClockTime now) const {
    double v = f.virtual_finish;
    if (cfg_.scheduling.deadline_boost_enabled) {
      std::int64_t slack = r.spec.deadline.valid() ? r.spec.deadline.remaining(now).count() : -1;
      if (slack >= 0 && slack <= cfg_.scheduling.deadline_boost_threshold_ms) v = v / cfg_.scheduling.deadline_boost_max;
    }
    return v;
  }
  // The head of every active flow, most deserving first: lowest effective
  // virtual finish, then flow key. Entries that are no longer queued are
  // dropped on the way, and heads whose deadline has passed are expired.
  std::vector<RequestId> runnable_heads(ClockTime now) {
    struct Head { double finish; FlowKey key; RequestId id; };
    std::vector<Head> heads;
    for (auto it = active_flows_.begin(); it != active_flows_.end();) {
      Flow& f = flows_[*it];
      bool alive = false;
      while (!f.pending.empty()) {
        ReadyEntry top = f.pending.top();
        auto rit = requests_.find(top.id);
        if (rit == requests_.end() || !queued_.count(top.id) || rit->second.generation != top.generation) { f.pending.pop(); continue; }
        RequestRecord& r = rit->second;
        if (r.spec.deadline.valid() && r.spec.deadline.expired(now)) { f.pending.pop(); expire_request(r); continue; }
        alive = true; break;
      }
      if (!alive) { auto rm = it++; active_flows_.erase(rm); continue; }
      const RequestRecord& r = requests_[f.pending.top().id];
      heads.push_back({effective_finish(f, r, now), *it, f.pending.top().id});
      ++it;
    }
    std::sort(heads.begin(), heads.end(), [](const Head& a, const Head& b) {
      if (a.finish != b.finish) return a.finish < b.finish;
      return a.key < b.key;
    });
    std::vector<RequestId> out;
    out.reserve(heads.size());
    for (const auto& h : heads) out.push_back(h.id);
    return out;
  }
  void account_flow(const RequestSpec& spec, double cost) {
    const FlowKey k = flow_key(spec);
    Flow& f = flows_[k];
    const double w = f.weight > 0.0 ? f.weight : 1.0;
    f.virtual_finish += cost / w;
    virtual_time_ = std::max(virtual_time_, f.virtual_finish);
  }
  std::vector<RequestId> form_batch(RequestId seed, ClockTime now) {
    std::vector<RequestId> result;
    auto sit = requests_.find(seed);
    if (sit == requests_.end()) return result;
    RequestRecord& s = sit->second;
    result.push_back(seed);
    int maxb = cfg_.batch.max_batch_size;
    if (s.spec.max_batch_size > 0) maxb = s.spec.max_batch_size;
    if (maxb < 1) maxb = 1;
    const CompatKey key = compat_key(s.spec);
    auto it = compat_.find(key);
    std::vector<RequestId> cands;
    if (it != compat_.end()) {
      for (auto id : it->second) {
        if (id == seed) continue;
        if (!queued_.count(id)) continue;
        auto c = requests_.find(id);
        if (c == requests_.end() || c->second.state != RequestState::Queued) continue;
        if (c->second.spec.deadline.valid() && c->second.spec.deadline.expired(now)) continue;
        cands.push_back(id);
      }
    }
    std::sort(cands.begin(), cands.end(), [&](RequestId a, RequestId b) {
      RequestRecord& ra = requests_[a]; RequestRecord& rb = requests_[b];
      std::int64_t sa = ra.spec.deadline.valid() ? ra.spec.deadline.remaining(now).count() : std::numeric_limits<std::int64_t>::max();
      std::int64_t sb = rb.spec.deadline.valid() ? rb.spec.deadline.remaining(now).count() : std::numeric_limits<std::int64_t>::max();
      if (sa != sb) return sa < sb;
      int pa = priority_int(ra.spec.priority); int pbb = priority_int(rb.spec.priority);
      if (pa != pbb) return pa > pbb;
      return a < b;
    });
    std::int64_t tok = s.spec.tokens.total();
    for (auto id : cands) {
      if (static_cast<int>(result.size()) >= maxb) break;
      const RequestSpec& cs = requests_[id].spec;
      // A member's own batch-size cap binds every batch it joins, not only the
      // ones it seeds.
      if (cs.max_batch_size > 0 && cs.max_batch_size < maxb) {
        if (static_cast<int>(result.size()) >= cs.max_batch_size) continue;
        maxb = cs.max_batch_size;
      }
      if (cfg_.batch.max_tokens > 0) { const std::int64_t add = cs.tokens.total(); if (add > cfg_.batch.max_tokens - tok) break; tok += add; }
      result.push_back(id);
    }
    return result;
  }

  // Units a worker can still take. What it reports as free is believed, but
  // never beyond what its registered capacity leaves once the work known to be
  // running there is counted; reservations not yet started come off as well.
  std::int64_t free_units(const WorkerRecord& wr) const {
    std::int64_t free = wr.resource.available_capacity_units;
    const std::int64_t total = wr.registration.capability.total_capacity_units;
    if (total > 0) free = std::min(free, total - wr.running_units);
    return free - wr.reserved_units;
  }
  // A worker's report of its free units is taken at its word, up to the
  // capacity it registered with.
  void apply_reported_capacity(WorkerRecord& wr, const ResourceSnapshot& snap) {
    if (snap.available_capacity_units < 0) return;
    const int total = wr.registration.capability.total_capacity_units;
    wr.resource.available_capacity_units = total > 0 ? std::min(snap.available_capacity_units, total) : snap.available_capacity_units;
    wr.resource.queue_depth = snap.queue_depth;
  }

  // ----- dispatch scoring -----
  WorkerScore score_worker(const WorkerRecord& wr, const RequestSpec& spec, double batch_cost, ClockTime now) const {
    WorkerScore s;
    s.worker = wr.registration.worker;
    const WorkerCapability& cap = wr.registration.capability;
    const auto& dw = cfg_.scheduling.dispatch;
    if (wr.state != WorkerState::Online && wr.state != WorkerState::Busy) { s.reject_reason_code = "worker_not_online"; return s; }
    if (!compatible(spec, cap)) { s.reject_reason_code = "incompatible"; return s; }
    const std::int64_t cost = ceil_cost(batch_cost);
    const std::int64_t avail = free_units(wr);
    if (cap.total_capacity_units > 0) {
      if (avail < cost) { s.reject_reason_code = "no_capacity"; return s; }
      if (cost > cap.total_capacity_units) { s.reject_reason_code = "oversized"; return s; }
    }
    if (spec.deadline.valid() && spec.deadline.expired(now)) { s.reject_reason_code = "deadline_expired"; return s; }
    s.eligible = true;
    double capfrac = cap.total_capacity_units > 0 ? static_cast<double>(avail) / static_cast<double>(cap.total_capacity_units) : 1.0;
    double load = 1.0;
    if (cap.total_capacity_units > 0) { const double busy = static_cast<double>(cap.total_capacity_units - avail); load = 1.0 - clamp01(busy / static_cast<double>(cap.total_capacity_units)); }
    double dl = 1.0;
    if (spec.deadline.valid()) { std::int64_t slack = spec.deadline.remaining(now).count(); dl = slack < 0 ? 0.0 : clamp01(static_cast<double>(slack) / static_cast<double>(std::max<std::int64_t>(1, cfg_.scheduling.deadline_boost_threshold_ms))); }
    double loc = 1.0;
    if (spec.locality.has_preference()) { loc = 0.0; if (spec.locality.accelerator.valid() && spec.locality.accelerator == wr.registration.accelerator) loc = 1.0; if (spec.locality.node.valid() && spec.locality.node == wr.registration.node) loc = std::max(loc, 0.8); }
    double warm = clamp01(spec.warmth.model_warmth * cap.model_warmth + 0.5 * (spec.warmth.model_warmth + cap.model_warmth) * 0.5);
    double state = clamp01(spec.warmth.state_warmth * cap.state_warmth + 0.5);
    const std::int64_t lat = static_cast<std::int64_t>(latency_int(spec.latency_class));
    const std::int64_t ph = static_cast<std::int64_t>(phase_int(spec.phase));
    s.components = {
      {"capability", 1.0, dw.capability},
      {"capacity", capfrac, dw.capacity},
      {"load", load, dw.load},
      {"deadline", dl, dw.deadline},
      {"latency_class", 1.0, dw.latency_class},
      {"phase", 1.0, dw.phase},
      {"locality", loc, dw.locality},
      {"model_warmth", warm, dw.model_warmth},
      {"state_warmth", state, dw.state_warmth},
      {"batch", 1.0, dw.batch},
      {"fairness", 1.0, dw.fairness}
    };
    (void)lat; (void)ph;
    double num = 0.0, den = 0.0;
    for (const auto& c : s.components) { if (c.weight > 0.0) { num += c.value * c.weight; den += c.weight; } }
    s.total = den > 0.0 ? num / den : 0.0;
    return s;
  }
  std::pair<WorkerId, std::vector<WorkerScore>> select_worker(const std::vector<RequestId>& members, double batch_cost, ClockTime now) const {
    std::vector<WorkerScore> scores;
    if (members.empty()) return {WorkerId(), scores};
    const RequestSpec& spec = requests_.at(members[0]).spec;
    for (const auto& [wk, wr] : workers_) {
      WorkerScore sc = score_worker(wr, spec, batch_cost, now);
      // Members share the seed's model, revision and adapter but, when phases
      // may mix, not its phase: each one has to be something this worker runs.
      for (std::size_t i = 1; sc.eligible && i < members.size(); ++i) {
        if (!compatible(requests_.at(members[i]).spec, wr.registration.capability)) {
          sc.eligible = false; sc.total = 0.0; sc.components.clear(); sc.reject_reason_code = "incompatible_member";
        }
      }
      scores.push_back(std::move(sc));
    }
    WorkerId best;
    double bestv = -std::numeric_limits<double>::infinity();
    bool found = false;
    for (const auto& sc : scores) {
      if (sc.eligible) {
        if (!found || sc.total > bestv || (sc.total == bestv && sc.worker < best)) { bestv = sc.total; best = sc.worker; found = true; }
      }
    }
    return {best, scores};
  }
  void release_batch(WorkerId worker, BatchId bid) {
    auto wit = workers_.find(worker); if (wit == workers_.end()) return;
    WorkerRecord& wr = wit->second;
    std::int64_t units = 0;
    auto bu = wr.batch_units.find(bid); if (bu != wr.batch_units.end()) units = bu->second;
    const bool started = wr.started_batches.count(bid) > 0;
    if (started) { wr.running_units -= units; global_running_units_ -= units; } else { wr.reserved_units -= units; global_reserved_units_ -= units; }
    wr.batch_units.erase(bid); wr.batch_pending.erase(bid); wr.batch_members.erase(bid); wr.started_batches.erase(bid);
    if (wr.running_units < 0) wr.running_units = 0;
    if (wr.reserved_units < 0) wr.reserved_units = 0;
    if (global_running_units_ < 0) global_running_units_ = 0;
    if (global_reserved_units_ < 0) global_reserved_units_ = 0;
    if (wr.running_units == 0 && wr.reserved_units == 0 && (wr.state == WorkerState::Busy)) wr.state = WorkerState::Online;
  }
  void release_batch_member(RequestRecord& rec, WorkerId worker) {
    auto wit = workers_.find(worker); if (wit == workers_.end()) return;
    WorkerRecord& wr = wit->second;
    wr.in_flight.erase(rec.request_id);
    if (rec.in_batch && rec.batch_id.valid()) {
      auto p = wr.batch_pending.find(rec.batch_id);
      if (p != wr.batch_pending.end()) { p->second -= 1; if (p->second <= 0) release_batch(worker, rec.batch_id); }
    }
    rec.in_batch = false; rec.batch_id = BatchId();
  }
  void finish_success(RequestRecord& rec, const CompletionReport& r) {
    rec.state = RequestState::Completing;
    rec.completed_at = clock_->now();
    rec.history.back().state = AttemptState::Succeeded;
    rec.history.back().final_status = CompletionStatus::Succeeded;
    rec.history.back().output_tokens = r.output_tokens_produced;
    rec.history.back().work_units = r.work_units;
    rec.history.back().duration_us = r.duration_us;
    rec.state = RequestState::Completed; rec.terminal_reason.clear();
    release_batch_member(rec, r.worker);
    release_admission(rec); leave_running(rec);
    stats_.requests_completed += 1; stats_.completions += 1;
    stats_.by_latency[latency_index(rec.spec.latency_class)].completed += 1;
    emit_event("completed", rec.request_id, rec.current_attempt, r.worker, rec.batch_id, rec.terminal_reason);
  }
  void finish_failed(RequestRecord& rec, const CompletionReport& r) {
    rec.history.back().state = AttemptState::Failed;
    rec.history.back().final_status = CompletionStatus::Failed;
    rec.history.back().failure_class = r.failure_class;
    rec.history.back().error_message = r.error_message;
    release_batch_member(rec, r.worker);
    rec.state = RequestState::Failed; rec.terminal_reason = name_of(r.failure_class);
    release_admission(rec); leave_running(rec);
    stats_.requests_failed += 1; stats_.completions += 1;
    emit_event("failed", rec.request_id, rec.current_attempt, r.worker, rec.batch_id, rec.terminal_reason);
  }
  void finish_cancelled(RequestRecord& rec, WorkerId w, std::string reason) {
    if (is_terminal(rec.state)) return;
    if (rec.state == RequestState::Reserved || rec.state == RequestState::Dispatched || rec.state == RequestState::Running) {
      release_batch_member(rec, w);
      leave_running(rec);
    }
    dequeue_request(rec);
    rec.cancellation_committed = true;
    rec.state = RequestState::Cancelled; rec.terminal_reason = std::move(reason);
    if (!rec.history.empty()) { rec.history.back().state = AttemptState::Cancelled; rec.history.back().final_status = CompletionStatus::Cancelled; }
    release_admission(rec);
    stats_.requests_cancelled += 1;
    emit_event("cancelled", rec.request_id, rec.current_attempt, w, rec.batch_id, rec.terminal_reason);
  }
  bool can_retry(const RequestRecord& rec, ClockTime now) const {
    if (rec.cancellation_requested || rec.cancellation_committed) return false;
    const int attempts = static_cast<int>(rec.history.size());
    if (attempts >= cfg_.retry.max_attempts) return false;
    if (rec.spec.deadline.valid() && rec.spec.deadline.expired(now)) return false;
    return true;
  }
  void make_retry(RequestRecord& rec, const CompletionReport& r) {
    rec.history.back().state = AttemptState::Failed;
    rec.history.back().final_status = CompletionStatus::Failed;
    rec.history.back().failure_class = r.failure_class;
    rec.history.back().error_message = r.error_message;
    release_batch_member(rec, r.worker);
    leave_running(rec);
    rec.generation = id_factory_.next<GenerationTag>();
    rec.current_attempt = id_factory_.next<AttemptIdTag>();
    AttemptRecord a; a.attempt_id = rec.current_attempt; a.generation = rec.generation; a.state = AttemptState::Created; a.created_at = clock_->now();
    rec.history.push_back(a);
    stats_.attempts_created += 1; stats_.retries += 1;
    rec.state = RequestState::Queued; rec.queued_at = clock_->now(); rec.in_batch = false; rec.batch_id = BatchId();
    enqueue_request(rec);
    emit_event("retry", rec.request_id, rec.current_attempt, r.worker, BatchId(), rec.terminal_reason);
  }

  // ----- dispatch planning -----
  Result<DispatchOutcome> do_plan_dispatch(RequestId request, bool explain) {
    std::lock_guard<std::mutex> g(mu_);
    const ClockTime now = clock_->now();
    expire_scan(now);
    return plan_dispatch_locked(request, explain, now);
  }
  Result<DispatchOutcome> plan_dispatch_locked(RequestId request, bool explain, ClockTime now) {
    auto it = requests_.find(request);
    if (it == requests_.end()) return Result<DispatchOutcome>::err(make_error(ErrorCode::NotFound, "request not found"));
    RequestRecord& rec = it->second;
    DispatchOutcome out; out.request_id = request; out.attempt_id = rec.current_attempt;
    if (rec.spec.deadline.valid() && rec.spec.deadline.expired(now)) {
      out.decision = DispatchDecision::Hold; out.reason_code = "deadline_expired";
      if (rec.state == RequestState::Queued) expire_request(rec);
      return Result<DispatchOutcome>::ok(std::move(out));
    }
    if (rec.state != RequestState::Queued && rec.state != RequestState::Admitted) {
      out.decision = DispatchDecision::Hold; out.reason_code = "not_runnable"; return Result<DispatchOutcome>::ok(std::move(out));
    }
    // batched members
    std::vector<RequestId> members = form_batch(request, now);
    if (members.empty()) members.push_back(request);
    std::int64_t batch_units = 0;
    std::pair<WorkerId, std::vector<WorkerScore>> sel;
    // A batch is only as large as some worker can take. When no worker has
    // room for all of it, the least urgent members stay queued rather than the
    // whole batch going nowhere. The units a worker is asked for are the units
    // that are then reserved on it.
    while (true) {
      batch_units = 0;
      for (auto id : members) batch_units += ceil_cost(requests_[id].spec.demand.cost_units);
      sel = select_worker(members, static_cast<double>(batch_units), now);
      if (sel.first.valid() || members.size() == 1) break;
      members.pop_back();
    }
    out.candidates = sel.second;
    if (!sel.first.valid()) {
      out.decision = DispatchDecision::NoEligibleWorker; out.reason_code = "no_eligible_worker";
      if (explain) out.explanation = "no eligible worker with capacity/compatibility";
      return Result<DispatchOutcome>::ok(std::move(out));
    }
    WorkerRecord& wr = workers_[sel.first];
    const std::int64_t avail = free_units(wr);
    if (wr.registration.capability.total_capacity_units > 0 && batch_units > avail) {
      out.decision = DispatchDecision::Hold; out.reason_code = "reservation_capacity_exceeded";
      return Result<DispatchOutcome>::ok(std::move(out));
    }
    // reserve batch
    const BatchId bid = id_factory_.next<BatchIdTag>();
    wr.reserved_units += batch_units; wr.batch_units[bid] = batch_units; wr.batch_pending[bid] = static_cast<int>(members.size()); wr.batch_members[bid] = members;
    global_reserved_units_ += batch_units;
    out.worker_id = sel.first; out.batch_id = bid; out.decision = DispatchDecision::Dispatch; out.reason_code = "dispatched";
    for (auto id : members) {
      RequestRecord& m = requests_[id];
      dequeue_request(m);
      account_flow(m.spec, std::max(1.0, m.spec.demand.cost_units));
      m.state = RequestState::Reserved; m.reserved_at = now; m.in_batch = true; m.batch_id = bid;
      // The attempt now belongs to this worker: only it may start or complete it.
      if (!m.history.empty()) { m.history.back().state = AttemptState::Reserved; m.history.back().worker = sel.first; m.history.back().batch = bid; }
      emit_event("reserved", m.request_id, m.current_attempt, sel.first, bid, m.terminal_reason);
    }
    out.estimated_cost_units = batch_units;
    out.batch_members = members;
    out.generation = requests_[request].generation;
    if (explain) { for (const auto& sc : sel.second) if (sc.worker == sel.first) out.explanation = "selected worker by weighted score"; }
    stats_.dispatches += 1; if (members.size() > 1) stats_.batches_formed += 1; stats_.batch_members += static_cast<std::uint64_t>(members.size());
    return Result<DispatchOutcome>::ok(std::move(out));
  }
  Result<DispatchOutcome> do_plan_next(bool explain) {
    std::lock_guard<std::mutex> g(mu_);
    const ClockTime now = clock_->now();
    expire_scan(now);
    const std::vector<RequestId> heads = runnable_heads(now);
    if (heads.empty()) { DispatchOutcome o; o.decision = DispatchDecision::Hold; o.reason_code = "no_runnable"; return Result<DispatchOutcome>::ok(std::move(o)); }
    // The most deserving flow goes first, but a head that no worker can take
    // right now does not hold up the flows behind it.
    Result<DispatchOutcome> first;
    bool have_first = false;
    for (const RequestId seed : heads) {
      Result<DispatchOutcome> r = plan_dispatch_locked(seed, explain, now);
      if (!r.ok() || r.value().decision == DispatchDecision::Dispatch) return r;
      if (!have_first) { first = std::move(r); have_first = true; }
    }
    return first;
  }
  Result<void> do_confirm_start(const CompletionReport& r) {
    std::lock_guard<std::mutex> g(mu_);
    if (r.epoch.value() != epoch_value_) return Result<void>::err(make_error(ErrorCode::StaleEpoch, "stale epoch"));
    auto it = requests_.find(r.request_id); if (it == requests_.end()) return Result<void>::err(make_error(ErrorCode::NotFound, "request not found"));
    RequestRecord& rec = it->second;
    if (rec.current_attempt != r.attempt_id || rec.generation != r.generation) return Result<void>::err(make_error(ErrorCode::StaleAttempt, "stale attempt"));
    if (rec.state != RequestState::Reserved) return Result<void>::err(make_error(ErrorCode::InvalidState, "not reserved"));
    auto wit = workers_.find(r.worker); if (wit == workers_.end() || wit->second.registration.boot_id != r.boot_id) return Result<void>::err(make_error(ErrorCode::StaleWorker, "stale worker"));
    // The attempt was reserved on one worker; no other worker can start it.
    if (rec.history.empty() || rec.history.back().worker != r.worker) return Result<void>::err(make_error(ErrorCode::StaleWorker, "attempt is reserved on another worker"));
    WorkerRecord& wr = wit->second;
    if (wr.state == WorkerState::Offline || wr.state == WorkerState::Unhealthy || wr.state == WorkerState::Drained) return Result<void>::err(make_error(ErrorCode::InvalidState, "worker not ready"));
    if (!rec.batch_id.valid()) return Result<void>::err(make_error(ErrorCode::InvalidState, "no batch"));
    if (wr.started_batches.count(rec.batch_id) == 0) {
      auto bu = wr.batch_units.find(rec.batch_id); std::int64_t units = (bu != wr.batch_units.end()) ? bu->second : 0;
      wr.reserved_units -= units; wr.running_units += units; global_reserved_units_ -= units; global_running_units_ += units;
      wr.started_batches.insert(rec.batch_id);
    }
    rec.state = RequestState::Running; rec.started_at = clock_->now();
    rec.schedule_delay_ms = ms_between(rec.queued_at, rec.started_at);
    rec.history.back().state = AttemptState::Running; rec.history.back().worker = r.worker; rec.history.back().batch = rec.batch_id;
    wr.in_flight[rec.request_id] = rec.current_attempt;
    auto& t = tenant_of(rec.spec.tenant); t.running += 1; rec.counted_running = true;
    if (wr.state == WorkerState::Online) wr.state = WorkerState::Busy;
    apply_reported_capacity(wr, r.worker_snapshot);
    emit_event("running", rec.request_id, rec.current_attempt, r.worker, rec.batch_id, "running");
    return Result<void>::success();
  }
  Result<CompletionOutcome> do_complete(const CompletionReport& r) {
    std::lock_guard<std::mutex> g(mu_);
    CompletionOutcome out;
    if (r.epoch.value() != epoch_value_) { out.acceptance = CompletionAcceptance::StaleEpoch; out.reason_code = "stale_epoch"; stats_.stale_rejected += 1; return Result<CompletionOutcome>::ok(std::move(out)); }
    auto wit = workers_.find(r.worker);
    if (wit == workers_.end() || wit->second.registration.boot_id != r.boot_id) { out.acceptance = CompletionAcceptance::StaleWorker; out.reason_code = "stale_worker"; stats_.stale_rejected += 1; return Result<CompletionOutcome>::ok(std::move(out)); }
    auto it = requests_.find(r.request_id); if (it == requests_.end()) { out.acceptance = CompletionAcceptance::Invalid; out.reason_code = "unknown_request"; return Result<CompletionOutcome>::ok(std::move(out)); }
    RequestRecord& rec = it->second;
    if (rec.current_attempt != r.attempt_id || rec.generation != r.generation) { out.acceptance = CompletionAcceptance::StaleAttempt; out.reason_code = "stale_attempt"; stats_.stale_rejected += 1; return Result<CompletionOutcome>::ok(std::move(out)); }
    if (is_terminal(rec.state)) { out.acceptance = CompletionAcceptance::Duplicate; out.reason_code = "duplicate_completion"; return Result<CompletionOutcome>::ok(std::move(out)); }
    if (rec.cancellation_committed) { out.acceptance = CompletionAcceptance::CancelledOutcome; out.reason_code = "cancelled"; return Result<CompletionOutcome>::ok(std::move(out)); }
    // An attempt can only be reported on once it has been handed to a worker,
    // and only by the worker it was handed to.
    if (rec.state != RequestState::Reserved && rec.state != RequestState::Dispatched && rec.state != RequestState::Running) { out.acceptance = CompletionAcceptance::Invalid; out.reason_code = "attempt_not_dispatched"; return Result<CompletionOutcome>::ok(std::move(out)); }
    if (rec.history.empty() || rec.history.back().worker != r.worker) { out.acceptance = CompletionAcceptance::StaleWorker; out.reason_code = "not_assigned_worker"; stats_.stale_rejected += 1; return Result<CompletionOutcome>::ok(std::move(out)); }
    // The report also carries the worker's capacity after the attempt.
    if (r.worker_snapshot.total_capacity_units > 0) apply_reported_capacity(wit->second, r.worker_snapshot);
    const ClockTime now = clock_->now();
    if (r.status == CompletionStatus::Succeeded) {
      if (rec.cancellation_requested) { finish_cancelled(rec, r.worker, "cancelled"); out.acceptance = CompletionAcceptance::CancelledOutcome; out.reason_code = "cancelled"; return Result<CompletionOutcome>::ok(std::move(out)); }
      finish_success(rec, r); out.acceptance = CompletionAcceptance::Accepted; out.reason_code = "success"; return Result<CompletionOutcome>::ok(std::move(out));
    }
    if (r.status == CompletionStatus::Cancelled) { finish_cancelled(rec, r.worker, "cancelled"); out.acceptance = CompletionAcceptance::Accepted; out.reason_code = "cancelled"; return Result<CompletionOutcome>::ok(std::move(out)); }
    if (r.status == CompletionStatus::Expired) { rec.history.back().state = AttemptState::Expired; release_batch_member(rec, r.worker); rec.state = RequestState::Expired; rec.terminal_reason = "deadline_expired"; release_admission(rec); leave_running(rec); stats_.requests_expired += 1; stats_.by_latency[latency_index(rec.spec.latency_class)].expired += 1; out.acceptance = CompletionAcceptance::Accepted; out.reason_code = "expired"; emit_event("expired", rec.request_id, rec.current_attempt, r.worker, rec.batch_id, rec.terminal_reason); return Result<CompletionOutcome>::ok(std::move(out)); }
    if (r.status == CompletionStatus::Failed || r.status == CompletionStatus::Ambiguous || r.status == CompletionStatus::Stale) {
      // A cancellation that was asked for while the attempt ran commits here,
      // whatever the attempt came to.
      if (rec.cancellation_requested) { finish_cancelled(rec, r.worker, "cancelled"); out.acceptance = CompletionAcceptance::CancelledOutcome; out.reason_code = "cancelled"; return Result<CompletionOutcome>::ok(std::move(out)); }
      if (r.status == CompletionStatus::Stale) {
        // The worker disowned the attempt. The request is not left without one:
        // it gets a new attempt, or fails when none is left.
        CompletionReport lost = r; lost.failure_class = FailureClass::StaleAuthority; lost.error_message = "worker reported the attempt stale";
        stats_.stale_rejected += 1;
        const std::size_t disowned = rec.history.size() - 1;
        if (can_retry(rec, now)) { make_retry(rec, lost); out.retried = true; out.retry_attempt = rec.current_attempt; } else { finish_failed(rec, lost); }
        rec.history[disowned].state = AttemptState::Stale;
        out.acceptance = CompletionAcceptance::StaleAttempt; out.reason_code = "stale"; return Result<CompletionOutcome>::ok(std::move(out));
      }
      const RetryPolicy& rp = cfg_.retry;
      bool retryable = (r.failure_class == FailureClass::RetryableTransport && rp.retry_on_transport) || (r.failure_class == FailureClass::RetryableWorker && rp.retry_on_worker_failure) || (r.failure_class == FailureClass::AmbiguousDispatch && rp.retry_on_ambiguous);
      if (retryable && can_retry(rec, now)) { make_retry(rec, r); out.retried = true; out.retry_attempt = rec.current_attempt; out.acceptance = CompletionAcceptance::Accepted; out.reason_code = "retry"; return Result<CompletionOutcome>::ok(std::move(out)); }
      finish_failed(rec, r); out.acceptance = CompletionAcceptance::Accepted; out.reason_code = "failed"; return Result<CompletionOutcome>::ok(std::move(out));
    }
    out.acceptance = CompletionAcceptance::Invalid; out.reason_code = "unrecognized_status"; return Result<CompletionOutcome>::ok(std::move(out));
  }

  // ----- cancellation / workers / inspection / drain -----
  Result<void> do_cancel(RequestId req, CancellationReason reason, std::string detail) {
    std::lock_guard<std::mutex> g(mu_);
    auto it = requests_.find(req); if (it == requests_.end()) return Result<void>::err(make_error(ErrorCode::NotFound, "request not found"));
    RequestRecord& rec = it->second;
    if (is_terminal(rec.state)) return Result<void>::err(make_error(ErrorCode::InvalidState, "already terminal"));
    rec.cancel_reason = reason;
    if (rec.state == RequestState::Reserved) { finish_cancelled(rec, rec.history.empty() ? WorkerId() : rec.history.back().worker, "cancelled"); return Result<void>::success(); }
    if (rec.state == RequestState::Dispatched || rec.state == RequestState::Running) {
      rec.cancellation_requested = true;
      emit_event("cancel_requested", req, rec.current_attempt, rec.history.back().worker, rec.batch_id, detail);
      return Result<void>::success();
    }
    finish_cancelled(rec, WorkerId(), "cancelled");
    return Result<void>::success();
  }
  void do_register_worker(const WorkerRegistration& reg) {
    std::lock_guard<std::mutex> g(mu_);
    WorkerRecord& wr = workers_[reg.worker];
    // If the worker id already exists under a different boot identity, any
    // in-flight work bound to the old identity is stale and must be recovered.
    if (wr.registration.worker.valid() && wr.registration.boot_id != reg.boot_id) recover_in_flight(reg.worker);
    wr.registration = reg;
    wr.state = (reg.state == WorkerState::Unknown || reg.state == WorkerState::Booted) ? WorkerState::Online : reg.state;
    wr.resource.total_capacity_units = reg.capability.total_capacity_units;
    wr.resource.available_capacity_units = reg.capability.available_capacity_units;
    wr.resource.memory_bytes_total = reg.capability.memory_bytes_total;
    wr.resource.memory_bytes_available = reg.capability.memory_bytes_available;
    if (wr.reserved_units < 0) wr.reserved_units = 0;
    emit_event("worker_registered", RequestId(), AttemptId(), reg.worker, BatchId(), reg.capability.backend);
  }
  // Everything a worker was holding is lost with it: each member of each batch
  // reserved or started there gets a new attempt, or fails when it has none
  // left, and the worker's reservations and running units go back to zero.
  void recover_in_flight(WorkerId worker, const char* why = "worker_restart") {
    auto wit = workers_.find(worker); if (wit == workers_.end()) return;
    WorkerRecord& wr = wit->second;
    std::vector<RequestId> ids;
    for (const auto& [bid, members] : wr.batch_members) for (auto rid : members) ids.push_back(rid);
    for (const auto& [rid, att] : wr.in_flight) ids.push_back(rid);
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    const ClockTime now = clock_->now();
    for (auto rid : ids) {
      auto rit = requests_.find(rid); if (rit == requests_.end()) continue;
      RequestRecord& rec = rit->second;
      if (rec.state != RequestState::Reserved && rec.state != RequestState::Dispatched && rec.state != RequestState::Running) continue;
      if (rec.history.empty() || rec.history.back().worker != worker) continue;
      CompletionReport rep; rep.worker = worker; rep.boot_id = wr.registration.boot_id; rep.request_id = rid; rep.attempt_id = rec.current_attempt; rep.generation = rec.generation; rep.status = CompletionStatus::Failed; rep.failure_class = FailureClass::RetryableWorker; rep.error_message = why;
      if (rec.cancellation_requested) { finish_cancelled(rec, worker, "cancelled"); }
      else if (cfg_.retry.retry_on_worker_failure && can_retry(rec, now)) { make_retry(rec, rep); }
      else { finish_failed(rec, rep); }
    }
    wr.in_flight.clear();
    // Whatever is still on the books belongs to requests that were already
    // terminal; it comes off the counter it was on.
    for (const auto& [bid, u] : wr.batch_units) {
      if (wr.started_batches.count(bid) > 0) { wr.running_units -= u; global_running_units_ -= u; }
      else { wr.reserved_units -= u; global_reserved_units_ -= u; }
    }
    wr.batch_units.clear(); wr.batch_pending.clear(); wr.batch_members.clear(); wr.started_batches.clear();
    if (wr.reserved_units < 0) wr.reserved_units = 0; if (wr.running_units < 0) wr.running_units = 0;
    if (global_reserved_units_ < 0) global_reserved_units_ = 0; if (global_running_units_ < 0) global_running_units_ = 0;
    if (wr.state == WorkerState::Busy) wr.state = WorkerState::Online;
  }
  Result<void> do_update_capability(WorkerId w, const WorkerCapability& cap) {
    std::lock_guard<std::mutex> g(mu_);
    auto it = workers_.find(w); if (it == workers_.end()) return Result<void>::err(make_error(ErrorCode::NotFound, "worker not found"));
    it->second.registration.capability = cap; it->second.resource.total_capacity_units = cap.total_capacity_units; it->second.resource.available_capacity_units = cap.available_capacity_units; it->second.resource.memory_bytes_total = cap.memory_bytes_total; it->second.resource.memory_bytes_available = cap.memory_bytes_available;
    emit_event("worker_updated", RequestId(), AttemptId(), w, BatchId(), "capability");
    return Result<void>::success();
  }
  Result<void> do_set_worker_state(WorkerId w, WorkerState state, const ResourceSnapshot* snap) {
    std::lock_guard<std::mutex> g(mu_);
    auto it = workers_.find(w); if (it == workers_.end()) return Result<void>::err(make_error(ErrorCode::NotFound, "worker not found"));
    it->second.state = state;
    if (snap) it->second.resource = *snap;
    emit_event("worker_state", RequestId(), AttemptId(), w, BatchId(), name_of(state));
    return Result<void>::success();
  }
  Result<void> do_unregister_worker(WorkerId w) {
    std::lock_guard<std::mutex> g(mu_);
    auto it = workers_.find(w); if (it == workers_.end()) return Result<void>::err(make_error(ErrorCode::NotFound, "worker not found"));
    recover_in_flight(w);
    workers_.erase(it);
    emit_event("worker_unregistered", RequestId(), AttemptId(), w, BatchId(), "unregistered");
    return Result<void>::success();
  }
  Result<WorkerSnapshot> do_worker_snapshot(WorkerId w) const {
    std::lock_guard<std::mutex> g(mu_);
    auto it = workers_.find(w); if (it == workers_.end()) return Result<WorkerSnapshot>::err(make_error(ErrorCode::NotFound, "worker not found"));
    const auto& wr = it->second;
    WorkerSnapshot ws; ws.worker = wr.registration.worker; ws.node = wr.registration.node; ws.boot_id = wr.registration.boot_id; ws.state = wr.state; ws.backend = wr.registration.capability.backend; ws.device_name = wr.registration.capability.device_name; ws.total_capacity_units = wr.registration.capability.total_capacity_units; ws.available_capacity_units = wr.resource.available_capacity_units; ws.reserved_units = static_cast<int>(wr.reserved_units); ws.running = static_cast<int>(wr.running_units); ws.memory_bytes_total = wr.resource.memory_bytes_total; ws.memory_bytes_available = wr.resource.memory_bytes_available;
    return Result<WorkerSnapshot>::ok(std::move(ws));
  }
  Result<RequestSnapshot> do_request_snapshot(RequestId id) const {
    std::lock_guard<std::mutex> g(mu_);
    auto it = requests_.find(id); if (it == requests_.end()) return Result<RequestSnapshot>::err(make_error(ErrorCode::NotFound, "request not found"));
    const auto& r = it->second;
    RequestSnapshot s; s.request_id = r.request_id; s.tenant = r.spec.tenant; s.state = r.state; s.phase = r.spec.phase; s.latency_class = r.spec.latency_class; s.priority = r.spec.priority; s.model = r.spec.model; s.revision = r.spec.revision; s.attempt_id = r.current_attempt; s.generation = r.generation; s.worker_id = r.history.empty() ? WorkerId() : r.history.back().worker; s.deadline = r.spec.deadline; s.queue_delay_ms = r.queue_delay_ms;
    return Result<RequestSnapshot>::ok(std::move(s));
  }
  SchedulerStats do_stats() const {
    std::lock_guard<std::mutex> g(mu_);
    SchedulerStats s = stats_;
    s.current_admitted = static_cast<int>(global_admitted_); s.current_queued = static_cast<int>(global_queued_);
    int reserved = 0, running = 0;
    for (const auto& [id, r] : requests_) { if (r.state == RequestState::Reserved || r.state == RequestState::Dispatched) ++reserved; if (r.state == RequestState::Running) ++running; }
    s.current_reserved = reserved; s.current_running = running;
    s.current_capacity_units = static_cast<int>(global_reserved_units_ + global_running_units_);
    s.current_reserved_units = static_cast<int>(global_reserved_units_);
    return s;
  }

  SchedulerSnapshot do_snapshot() const {
    std::lock_guard<std::mutex> g(mu_);
    SchedulerSnapshot ss;
    ss.epoch = SchedulerEpoch(epoch_value_);
    ss.generated_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(clock_->now().time_since_epoch()).count();
    ss.stats = do_stats_locked();
    for (const auto& [tid, t] : tenants_) {
      TenantSnapshot ts; ts.tenant = tid; ts.weight = t.weight; ts.queued = t.queued; ts.admitted = t.admitted; ts.services_consumed = t.services_consumed;
      ts.virtual_time = 0.0; ts.service_debt = t.service_debt; ss.tenants.push_back(ts);
    }
    for (const auto& [wk, wr] : workers_) {
      WorkerSnapshot ws; ws.worker = wr.registration.worker; ws.node = wr.registration.node; ws.boot_id = wr.registration.boot_id; ws.state = wr.state; ws.backend = wr.registration.capability.backend; ws.device_name = wr.registration.capability.device_name; ws.total_capacity_units = wr.registration.capability.total_capacity_units; ws.available_capacity_units = wr.resource.available_capacity_units; ws.reserved_units = static_cast<int>(wr.reserved_units); ws.running = static_cast<int>(wr.running_units); ws.memory_bytes_total = wr.resource.memory_bytes_total; ws.memory_bytes_available = wr.resource.memory_bytes_available; ss.workers.push_back(ws);
    }
    std::vector<RequestId> ids;
    for (const auto& [id, r] : requests_) { if (!is_terminal(r.state)) ids.push_back(id); }
    std::sort(ids.begin(), ids.end());
    for (auto id : ids) { const auto& r = requests_.at(id); RequestSnapshot rs; rs.request_id = id; rs.tenant = r.spec.tenant; rs.state = r.state; rs.phase = r.spec.phase; rs.latency_class = r.spec.latency_class; rs.priority = r.spec.priority; rs.model = r.spec.model; rs.revision = r.spec.revision; rs.attempt_id = r.current_attempt; rs.generation = r.generation; rs.worker_id = r.history.empty() ? WorkerId() : r.history.back().worker; rs.deadline = r.spec.deadline; rs.queue_delay_ms = r.queue_delay_ms; ss.requests.push_back(rs); }
    return ss;
  }
  SchedulerStats do_stats_locked() const {
    SchedulerStats s = stats_;
    s.current_admitted = static_cast<int>(global_admitted_); s.current_queued = static_cast<int>(global_queued_);
    int reserved = 0, running = 0;
    for (const auto& [id, r] : requests_) { if (r.state == RequestState::Reserved || r.state == RequestState::Dispatched) ++reserved; if (r.state == RequestState::Running) ++running; }
    s.current_reserved = reserved; s.current_running = running;
    s.current_capacity_units = static_cast<int>(global_reserved_units_ + global_running_units_);
    s.current_reserved_units = static_cast<int>(global_reserved_units_);
    return s;
  }
  ExplainResult do_explain(RequestId id) const {
    std::lock_guard<std::mutex> g(mu_);
    ExplainResult ex;
    ex.epoch = SchedulerEpoch(epoch_value_);
    auto it = requests_.find(id);
    if (it == requests_.end()) { ex.notes.push_back("unknown request"); return ex; }
    const RequestRecord& rec = it->second;
    ex.request_id = id; ex.attempt_id = rec.current_attempt;
    const ClockTime now = clock_->now();
    if (is_terminal(rec.state)) {
      ex.admission = rec.state == RequestState::Rejected ? AdmissionDecision::Reject : AdmissionDecision::Admit;
      ex.admitted = false; ex.admission_reason_code = "terminal:" + std::string(name_of(rec.state));
      ex.notes.push_back("request is terminal: " + std::string(name_of(rec.state)));
      return ex;
    }
    // A request that is on record and not terminal was admitted. Running
    // admission again now would judge it against limits it is itself counted in.
    ex.admission = AdmissionDecision::Admit; ex.admission_reason_code = "admitted"; ex.admitted = true;
    ex.admission_explanation = "admitted";
    const FlowKey k = flow_key(rec.spec);
    auto fit = flows_.find(k);
    if (fit != flows_.end()) {
      ex.ordering_factors.push_back("flow_weight=" + std::to_string(fit->second.weight));
      ex.ordering_factors.push_back("virtual_finish=" + std::to_string(fit->second.virtual_finish));
    }
    std::int64_t slack = rec.spec.deadline.valid() ? rec.spec.deadline.remaining(now).count() : -1;
    ex.ordering_factors.push_back("deadline_slack_ms=" + std::to_string(slack));
    ex.ordering_factors.push_back("latency_class=" + std::string(name_of(rec.spec.latency_class)));
    ex.ordering_factors.push_back("priority=" + std::string(name_of(rec.spec.priority)));
    ex.ordering_factors.push_back("phase=" + std::string(name_of(rec.spec.phase)));
    ex.ordering_factors.push_back("kept_at_deadline=" + std::to_string(1));
    const double cost = std::max(1.0, rec.spec.demand.cost_units);
    for (const auto& [wk, wr] : workers_) { ex.candidates.push_back(score_worker(wr, rec.spec, cost, now)); }
    auto sel = select_worker({id}, cost, now);
    ex.selected_worker = sel.first;
    for (const auto& sc : sel.second) if (sc.worker == sel.first && sc.eligible) ex.notes.push_back("selected worker " + sel.first.str());
    return ex;
  }
  Result<void> do_begin_drain(std::string reason) {
    std::lock_guard<std::mutex> g(mu_);
    draining_ = true; drain_reason_ = std::move(reason);
    emit_event("drain_started", RequestId(), AttemptId(), WorkerId(), BatchId(), drain_reason_);
    return Result<void>::success();
  }
  Result<void> do_cancel_all_queued(std::string detail) {
    std::lock_guard<std::mutex> g(mu_);
    std::vector<RequestId> q(queued_.begin(), queued_.end());
    for (auto id : q) { auto it = requests_.find(id); if (it != requests_.end() && !is_terminal(it->second.state)) { RequestRecord& rr = it->second; rr.cancel_reason = CancellationReason::SchedulerDrain; finish_cancelled(rr, WorkerId(), detail.empty() ? "scheduler_drain" : detail); } }
    return Result<void>::success();
  }
  void do_roll_epoch() {
    std::lock_guard<std::mutex> g(mu_);
    epoch_value_ += 1;
    // Attempts issued under the old epoch can no longer be completed: nothing a
    // worker reports for them is accepted from here on. They are recovered now
    // instead of being left reserved or running for good.
    std::vector<WorkerId> ids;
    for (const auto& [wk, wr] : workers_) ids.push_back(wk);
    std::sort(ids.begin(), ids.end());
    for (const WorkerId w : ids) recover_in_flight(w, "epoch_rollover");
    emit_event("epoch_rollover", RequestId(), AttemptId(), WorkerId(), BatchId(), "epoch=" + std::to_string(epoch_value_));
  }
  Result<void> do_persist(const std::string& path) const {
    std::lock_guard<std::mutex> g(mu_);
    if (!cfg_.persist_recovery) return Result<void>::err(make_error(ErrorCode::InvalidState, "persistence disabled"));
    std::string payload = serialize_locked();
    return detail::write_persistence_file(path, payload);
  }
  Result<void> do_recover(const std::string& path) {
    std::lock_guard<std::mutex> g(mu_);
    auto payload = detail::read_persistence_file(path);
    if (!payload) return Result<void>::err(payload.error());
    recovery_mode_ = 1;
    auto res = deserialize_locked(payload.value());
    recovery_mode_ = 0;
    return res;
  }

  void emit_event(const std::string& type, RequestId rid, AttemptId aid, WorkerId wid, BatchId bid, const std::string& detail) {
    SchedulerEvent e;
    e.type = type; e.request_id = rid; e.attempt_id = aid; e.worker_id = wid; e.batch_id = bid; e.detail = detail;
    e.epoch = SchedulerEpoch(epoch_value_);
    e.timestamp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(clock_->now().time_since_epoch()).count();
    std::lock_guard<std::mutex> g(event_mu_);
    e.sequence = ++event_seq_;
    event_log_.push_back(std::move(e));
    if (event_log_.size() > 65536u) event_log_.pop_front();
  }
  std::vector<SchedulerEvent> drain_events() {
    std::vector<SchedulerEvent> out;
    { std::lock_guard<std::mutex> g(event_mu_); out.assign(event_log_.begin(), event_log_.end()); event_log_.clear(); }
    return out;
  }
  void set_event_handler(std::function<void(const SchedulerEvent&)> h) {
    std::lock_guard<std::mutex> g(event_mu_);
    event_handler_ = std::move(h); event_stream_mode_ = static_cast<bool>(event_handler_);
  }
  void flush_events() {
    std::function<void(const SchedulerEvent&)> h;
    std::deque<SchedulerEvent> batch;
    { std::lock_guard<std::mutex> g(event_mu_); if (!event_stream_mode_) return; h = event_handler_; batch.swap(event_log_); }
    if (h) { for (const auto& e : batch) h(e); }
  }
  std::string serialize_locked() const {
    BinWriter w;
    w.u32(0x49534348u); w.u32(1u);
    w.u64(epoch_value_);
    w.i64(global_admitted_); w.i64(global_queued_); w.i64(global_reserved_units_); w.i64(global_running_units_); w.i64(global_tokens_);
    w.u64(stats_.requests_submitted); w.u64(stats_.requests_admitted); w.u64(stats_.requests_completed); w.u64(stats_.requests_cancelled); w.u64(stats_.requests_failed); w.u64(stats_.requests_expired); w.u64(stats_.attempts_created); w.u64(stats_.dispatches); w.u64(stats_.retries);
    w.u64(requests_.size());
    for (const auto& [rid, r] : requests_) {
      w.id(rid); w.i32(static_cast<std::int32_t>(r.state)); w.id(r.current_attempt); w.id(r.generation); w.i32(r.attempts_used);
      w.yes(r.cancellation_requested); w.yes(r.cancellation_committed); w.i32(static_cast<std::int32_t>(r.cancel_reason)); w.str(r.terminal_reason);
      w.id(r.spec.tenant); w.id(r.spec.session); w.id(r.spec.sequence); w.id(r.spec.model); w.id(r.spec.revision); w.opt_id(r.spec.adapter);
      w.i32(static_cast<std::int32_t>(r.spec.phase)); w.i32(priority_int(r.spec.priority)); w.i32(latency_int(r.spec.latency_class));
      w.i64(r.spec.tokens.input); w.i64(r.spec.tokens.output); w.i64(r.spec.tokens.decoded);
      w.i64(r.spec.estimated_memory_bytes); w.i32(r.spec.max_batch_size); w.str(r.spec.model_name);
      w.i64(r.spec.demand.memory_bytes); w.dbl(r.spec.demand.cost_units);
      w.i64(r.spec.deadline.valid() ? r.spec.deadline.remaining(clock_->now()).count() : -1);
      w.u64(r.history.size());
      for (const auto& a : r.history) { w.id(a.attempt_id); w.id(a.generation); w.i32(static_cast<std::int32_t>(a.state)); w.id(a.worker); w.id(a.batch); w.i32(static_cast<std::int32_t>(a.final_status)); w.i32(static_cast<std::int32_t>(a.failure_class)); w.str(a.error_message); w.i64(a.output_tokens); w.i64(a.work_units); w.i64(a.duration_us); }
    }
    w.u64(tenants_.size());
    for (const auto& [tid, t] : tenants_) { w.id(tid); w.dbl(t.weight); w.i32(t.admitted); w.i32(t.queued); w.i32(t.running); w.i64(t.services_consumed); w.i64(t.service_debt); }
    w.u64(workers_.size());
    for (const auto& [wk, wr] : workers_) {
      w.id(wr.registration.worker); w.id(wr.registration.node); w.id(wr.registration.accelerator); w.id(wr.registration.boot_id);
      w.i32(static_cast<std::int32_t>(wr.state)); w.str(wr.registration.capability.backend); w.str(wr.registration.capability.device_name);
      w.i32(wr.registration.capability.total_capacity_units); w.i32(wr.registration.capability.available_capacity_units);
      w.i64(wr.registration.capability.memory_bytes_total); w.i64(wr.registration.capability.memory_bytes_available);
      w.u64(wr.registration.capability.models.size()); for (const auto& x : wr.registration.capability.models) w.id(x);
      w.u64(wr.registration.capability.revisions.size()); for (const auto& x : wr.registration.capability.revisions) w.id(x);
      w.u64(wr.registration.capability.adapters.size()); for (const auto& x : wr.registration.capability.adapters) w.id(x);
      w.u64(wr.registration.capability.phases.size()); for (const auto& x : wr.registration.capability.phases) w.i32(static_cast<std::int32_t>(x));
    }
    return w.buf;
  }
  Result<void> deserialize_locked(std::string_view data) {
    const auto corrupt = [](const char* what) { return Result<void>::err(make_error(ErrorCode::Corruption, what)); };
    BinReader r; r.s = data;
    std::uint32_t magic = r.u32(); std::uint32_t ver = r.u32();
    if (!r.ok || magic != 0x49534348u) return corrupt("bad magic");
    if (ver != 1u) return corrupt("bad format version");
    // Everything is read and checked into staged containers first: a payload
    // that turns out to be bad leaves the live scheduler exactly as it was.
    std::unordered_map<RequestId, RequestRecord> requests;
    std::unordered_map<TenantId, TenantRecord> tenants;
    std::unordered_map<WorkerId, WorkerRecord> workers;
    const std::uint64_t epoch = r.u64();
    // The counters stored next all follow from the request records. They are
    // rebuilt from those records below rather than trusted from the file.
    for (int i = 0; i < 5; ++i) (void)r.i64();
    std::uint64_t st[9] = {};
    for (auto& v : st) v = r.u64();
    std::uint64_t nreq = r.u64(); if (!r.ok || nreq > 10000000u) return corrupt("bad request count");
    const ClockTime now = clock_->now();
    for (std::uint64_t i = 0; i < nreq; ++i) {
      RequestRecord rec;
      rec.request_id = r.id<RequestIdTag>(); const std::int32_t state = r.i32(); rec.current_attempt = r.id<AttemptIdTag>(); rec.generation = r.id<GenerationTag>(); rec.attempts_used = r.i32();
      if (!r.ok) return corrupt("request record truncated");
      rec.cancellation_requested = r.yes(); rec.cancellation_committed = r.yes(); const std::int32_t cancel_reason = r.i32(); rec.terminal_reason = r.str();
      rec.spec.tenant = r.id<TenantIdTag>(); rec.spec.session = r.id<SessionIdTag>(); rec.spec.sequence = r.id<SequenceIdTag>(); rec.spec.model = r.id<ModelIdentityTag>(); rec.spec.revision = r.id<ModelRevisionTag>(); r.opt_id(rec.spec.adapter);
      const std::int32_t phase = r.i32(); const std::int32_t priority = r.i32(); const std::int32_t latency = r.i32();
      rec.spec.tokens.input = r.i64(); rec.spec.tokens.output = r.i64(); rec.spec.tokens.decoded = r.i64();
      rec.spec.estimated_memory_bytes = r.i64(); rec.spec.max_batch_size = r.i32(); rec.spec.model_name = r.str();
      rec.spec.demand.memory_bytes = r.i64(); rec.spec.demand.cost_units = r.dbl();
      std::int64_t rem = r.i64();
      std::uint64_t nh = r.u64(); if (!r.ok || nh > 4096u) return corrupt("bad history count");
      if (!enum_in_range(state, RequestState::Failed) || !enum_in_range(cancel_reason, CancellationReason::Manual) ||
          !enum_in_range(phase, RequestPhase::Decode) || !enum_in_range(priority, PriorityClass::Critical) ||
          !enum_in_range(latency, LatencyClass::Interactive)) return corrupt("request record has an impossible enum value");
      rec.state = static_cast<RequestState>(state); rec.cancel_reason = static_cast<CancellationReason>(cancel_reason);
      rec.spec.phase = static_cast<RequestPhase>(phase); rec.spec.priority = static_cast<PriorityClass>(priority); rec.spec.latency_class = static_cast<LatencyClass>(latency);
      if (!rec.request_id.valid() || rem < -1 || !validate_spec(rec.spec)) return corrupt("request record is not a valid request");
      // A deadline is stored as the time it had left; one further out than the
      // clock can represent is as far out as the clock goes.
      rem = std::min<std::int64_t>(rem, 4000000000000LL);
      if (rem >= 0) rec.spec.deadline = Deadline::after(std::chrono::milliseconds(rem), now);
      for (std::uint64_t j = 0; j < nh; ++j) {
        AttemptRecord a; a.attempt_id = r.id<AttemptIdTag>(); a.generation = r.id<GenerationTag>(); const std::int32_t astate = r.i32(); a.worker = r.id<WorkerIdTag>(); a.batch = r.id<BatchIdTag>(); const std::int32_t fstatus = r.i32(); const std::int32_t fclass = r.i32(); a.error_message = r.str(); a.output_tokens = r.i64(); a.work_units = r.i64(); a.duration_us = r.i64();
        if (!r.ok) return corrupt("attempt record truncated");
        if (!enum_in_range(astate, AttemptState::Expired) || !enum_in_range(fstatus, CompletionStatus::Ambiguous) || !enum_in_range(fclass, FailureClass::Internal)) return corrupt("attempt record has an impossible enum value");
        a.state = static_cast<AttemptState>(astate); a.final_status = static_cast<CompletionStatus>(fstatus); a.failure_class = static_cast<FailureClass>(fclass);
        rec.history.push_back(std::move(a));
      }
      // A request that can still run is identified by its latest attempt.
      if (!is_terminal(rec.state) && !rec.history.empty() &&
          (rec.history.back().attempt_id != rec.current_attempt || rec.history.back().generation != rec.generation)) return corrupt("request does not match its latest attempt");
      rec.created_at = now; rec.completed_at = now;
      const RequestId rid = rec.request_id;
      if (!requests.insert({rid, std::move(rec)}).second) return corrupt("duplicate request id");
    }
    std::uint64_t nten = r.u64(); if (!r.ok || nten > 100000u) return corrupt("bad tenant count");
    for (std::uint64_t i = 0; i < nten; ++i) {
      TenantRecord t; t.tenant = r.id<TenantIdTag>(); t.weight = r.dbl(); (void)r.i32(); (void)r.i32(); (void)r.i32(); t.services_consumed = r.i64(); t.service_debt = r.i64();
      if (!r.ok) return corrupt("tenant record corrupted");
      tenants.insert({t.tenant, t});
    }
    std::uint64_t nwrk = r.u64(); if (!r.ok || nwrk > 100000u) return corrupt("bad worker count");
    for (std::uint64_t i = 0; i < nwrk; ++i) {
      WorkerRecord wr;
      wr.registration.worker = r.id<WorkerIdTag>(); wr.registration.node = r.id<NodeIdTag>(); wr.registration.accelerator = r.id<AcceleratorIdTag>(); wr.registration.boot_id = r.id<WorkerBootIdTag>();
      const std::int32_t wstate = r.i32(); wr.registration.capability.backend = r.str(); wr.registration.capability.device_name = r.str();
      wr.registration.capability.total_capacity_units = r.i32(); wr.registration.capability.available_capacity_units = r.i32();
      wr.registration.capability.memory_bytes_total = r.i64(); wr.registration.capability.memory_bytes_available = r.i64();
      std::uint64_t nm = r.u64(); if (!r.ok || nm > 4096u) return corrupt("bad model count"); for (std::uint64_t j = 0; j < nm; ++j) wr.registration.capability.models.push_back(r.id<ModelIdentityTag>());
      std::uint64_t nrv = r.u64(); if (!r.ok || nrv > 4096u) return corrupt("bad revision count"); for (std::uint64_t j = 0; j < nrv; ++j) wr.registration.capability.revisions.push_back(r.id<ModelRevisionTag>());
      std::uint64_t nad = r.u64(); if (!r.ok || nad > 4096u) return corrupt("bad adapter count"); for (std::uint64_t j = 0; j < nad; ++j) wr.registration.capability.adapters.push_back(r.id<AdapterIdentityTag>());
      std::uint64_t nph = r.u64(); if (!r.ok || nph > 16u) return corrupt("bad phase count");
      for (std::uint64_t j = 0; j < nph; ++j) { const std::int32_t ph = r.i32(); if (!enum_in_range(ph, RequestPhase::Decode)) return corrupt("worker record has an impossible enum value"); wr.registration.capability.phases.push_back(static_cast<RequestPhase>(ph)); }
      if (!r.ok) return corrupt("worker record corrupted");
      if (!enum_in_range(wstate, WorkerState::Unhealthy) || !wr.registration.worker.valid()) return corrupt("worker record has an impossible value");
      wr.state = static_cast<WorkerState>(wstate);
      wr.resource.total_capacity_units = wr.registration.capability.total_capacity_units; wr.resource.available_capacity_units = wr.registration.capability.available_capacity_units; wr.resource.memory_bytes_total = wr.registration.capability.memory_bytes_total; wr.resource.memory_bytes_available = wr.registration.capability.memory_bytes_available;
      const WorkerId wid = wr.registration.worker;
      if (!workers.insert({wid, std::move(wr)}).second) return corrupt("duplicate worker id");
    }
    if (!r.ok || r.off != data.size()) return corrupt("trailing corruption at end of payload");

    // The payload is sound: it replaces the live state.
    requests_.swap(requests); tenants_.swap(tenants); workers_.swap(workers);
    flows_.clear(); compat_.clear(); active_flows_.clear(); queued_.clear();
    epoch_value_ = epoch;
    stats_.requests_submitted = st[0]; stats_.requests_admitted = st[1]; stats_.requests_completed = st[2]; stats_.requests_cancelled = st[3]; stats_.requests_failed = st[4]; stats_.requests_expired = st[5]; stats_.attempts_created = st[6]; stats_.dispatches = st[7]; stats_.retries = st[8];
    global_admitted_ = 0; global_queued_ = 0; global_reserved_units_ = 0; global_running_units_ = 0; global_tokens_ = 0; active_tenants_ = 0;
    std::vector<RequestId> live;
    for (const auto& [id, rec] : requests_) { if (!is_terminal(rec.state)) live.push_back(id); }
    std::sort(live.begin(), live.end());
    for (auto id : live) {
      RequestRecord& rec = requests_[id];
      const bool handed_out = rec.state == RequestState::Reserved || rec.state == RequestState::Dispatched || rec.state == RequestState::Running || rec.state == RequestState::Completing;
      if (handed_out) {
        // Work that was reserved or running when the state was written cannot
        // be proven alive, and no reservation survives a restart. It comes back
        // as a new attempt, so a late report for the old one is stale; with a
        // cancellation pending or no attempt left it ends here instead.
        if (!rec.history.empty()) { rec.history.back().state = AttemptState::Stale; rec.history.back().final_status = CompletionStatus::Ambiguous; rec.history.back().failure_class = FailureClass::AmbiguousDispatch; }
        if (rec.cancellation_requested) { rec.state = RequestState::Cancelled; rec.cancellation_committed = true; rec.terminal_reason = "cancelled"; stats_.requests_cancelled += 1; continue; }
        if (static_cast<int>(rec.history.size()) >= cfg_.retry.max_attempts) { rec.state = RequestState::Failed; rec.terminal_reason = "recovery_attempts_exhausted"; stats_.requests_failed += 1; continue; }
        rec.generation = id_factory_.next<GenerationTag>(); rec.current_attempt = id_factory_.next<AttemptIdTag>();
        AttemptRecord a; a.attempt_id = rec.current_attempt; a.generation = rec.generation; a.state = AttemptState::Created; a.created_at = now;
        rec.history.push_back(a);
        stats_.attempts_created += 1; stats_.retries += 1;
      }
      rec.in_batch = false; rec.batch_id = BatchId(); rec.counted_running = false;
      hold_admission(rec.spec);
      enqueue_locked(rec);
    }
    return Result<void>::success();
  }
  void enqueue_locked(const RequestRecord& rcref) {
    RequestRecord& rec = requests_[rcref.request_id];
    rec.state = RequestState::Queued;
    const FlowKey k = flow_key(rec.spec); Flow& f = flows_[k]; f.weight = flow_weight(rec.spec);
    const ClockTime now = clock_->now();
    ReadyEntry e; e.id = rec.request_id; e.generation = rec.generation; e.slack = rec.spec.deadline.valid() ? rec.spec.deadline.remaining(now).count() : std::numeric_limits<std::int64_t>::max(); e.prio = priority_int(rec.spec.priority); e.idv = rec.request_id.value();
    if (f.pending.empty()) { f.virtual_finish = std::max(f.virtual_finish, virtual_time_); active_flows_.insert(k); }
    f.pending.push(e); queued_.insert(rec.request_id); compat_[compat_key(rec.spec)].insert(rec.request_id);
    global_queued_ += 1; tenant_of(rec.spec.tenant).queued += 1;
  }
  Result<RequestSpec> do_request_spec(RequestId id) const { std::lock_guard<std::mutex> g(mu_); auto it = requests_.find(id); if (it == requests_.end()) return Result<RequestSpec>::err(make_error(ErrorCode::NotFound, "request not found")); return Result<RequestSpec>::ok(it->second.spec); }
  std::size_t do_queued_count() const { std::lock_guard<std::mutex> g(mu_); return static_cast<std::size_t>(global_queued_ < 0 ? 0 : global_queued_); }
  std::size_t do_reserved_count() const { std::lock_guard<std::mutex> g(mu_); std::size_t c = 0; for (const auto& [id, r] : requests_) { if (r.state == RequestState::Reserved || r.state == RequestState::Dispatched) ++c; } return c; }
  int do_available_capacity() const { std::lock_guard<std::mutex> g(mu_); std::int64_t s = 0; for (const auto& [wk, wr] : workers_) { s += std::max<std::int64_t>(0, free_units(wr)); } const std::int64_t top = std::numeric_limits<int>::max(); return static_cast<int>(s > top ? top : s); }
};

// Scheduler wrapper methods
Scheduler::Scheduler(SchedulerConfig config) : Scheduler(std::move(config), nullptr, std::nullopt) {}
Scheduler::Scheduler(SchedulerConfig config, std::shared_ptr<Clock> clock) : Scheduler(std::move(config), std::move(clock), std::nullopt) {}
Scheduler::Scheduler(SchedulerConfig config, std::shared_ptr<Clock> clock, std::optional<SchedulerEpoch> epoch)
    : impl_(std::make_unique<Scheduler::Impl>(std::move(config), std::move(clock), std::move(epoch))) {}
Scheduler::~Scheduler() = default;

SchedulerConfig Scheduler::config() const { return impl_->cfg_; }
SchedulerEpoch Scheduler::epoch() const {
  // roll_epoch() writes the epoch under the state mutex; read it the same way.
  std::lock_guard<std::mutex> g(impl_->mu_);
  return SchedulerEpoch(impl_->epoch_value_);
}
std::shared_ptr<Clock> Scheduler::clock() const { return impl_->clock_; }

Result<AdmissionOutput> Scheduler::submit(const RequestSpec& spec, std::optional<RequestId> pre) {
  auto r = impl_->do_submit(spec, pre); impl_->flush_events(); return r;
}
Result<AdmissionOutput> Scheduler::admit(const RequestSpec& spec, std::optional<RequestId> pre) {
  auto r = impl_->do_submit(spec, pre); impl_->flush_events(); return r;
}
Result<void> Scheduler::cancel(RequestId request, CancellationReason reason, std::string detail) {
  auto r = impl_->do_cancel(request, reason, std::move(detail)); impl_->flush_events(); return r;
}

void Scheduler::register_worker(const WorkerRegistration& registration) {
  impl_->do_register_worker(registration); impl_->flush_events();
}
Result<void> Scheduler::update_worker_capability(WorkerId worker, const WorkerCapability& capability) { return impl_->do_update_capability(worker, capability); }
Result<void> Scheduler::set_worker_state(WorkerId worker, WorkerState state, const ResourceSnapshot* snapshot) {
  auto r = impl_->do_set_worker_state(worker, state, snapshot); impl_->flush_events(); return r;
}
Result<void> Scheduler::unregister_worker(WorkerId worker) {
  auto r = impl_->do_unregister_worker(worker); impl_->flush_events(); return r;
}
Result<WorkerSnapshot> Scheduler::worker_snapshot(WorkerId worker) const { return impl_->do_worker_snapshot(worker); }

Result<DispatchOutcome> Scheduler::plan_dispatch(RequestId request, bool include_explanation) {
  auto r = impl_->do_plan_dispatch(request, include_explanation); impl_->flush_events(); return r;
}
Result<DispatchOutcome> Scheduler::plan_next_dispatch(bool include_explanation) {
  auto r = impl_->do_plan_next(include_explanation); impl_->flush_events(); return r;
}
Result<void> Scheduler::confirm_dispatch_start(const CompletionReport& report) {
  auto r = impl_->do_confirm_start(report); impl_->flush_events(); return r;
}
Result<CompletionOutcome> Scheduler::complete_attempt(const CompletionReport& report) {
  auto r = impl_->do_complete(report); impl_->flush_events(); return r;
}

Result<RequestSpec> Scheduler::request_spec(RequestId request) const { return impl_->do_request_spec(request); }
Result<RequestSnapshot> Scheduler::request_snapshot(RequestId request) const { return impl_->do_request_snapshot(request); }
SchedulerStats Scheduler::stats() const { return impl_->do_stats(); }
SchedulerSnapshot Scheduler::snapshot() const { return impl_->do_snapshot(); }
std::vector<SchedulerEvent> Scheduler::drain_events() { return impl_->drain_events(); }
void Scheduler::set_event_handler(std::function<void(const SchedulerEvent&)> handler) { impl_->set_event_handler(std::move(handler)); }
std::size_t Scheduler::queued_count() const { return impl_->do_queued_count(); }
std::size_t Scheduler::reserved_count() const { return impl_->do_reserved_count(); }
int Scheduler::available_worker_capacity_units() const { return impl_->do_available_capacity(); }

Result<ExplainResult> Scheduler::explain(RequestId request) const { return impl_->do_explain(request); }

Result<void> Scheduler::begin_drain(std::string reason) { auto r = impl_->do_begin_drain(std::move(reason)); impl_->flush_events(); return r; }
Result<void> Scheduler::cancel_all_queued(std::string detail) { auto r = impl_->do_cancel_all_queued(std::move(detail)); impl_->flush_events(); return r; }
Result<void> Scheduler::recover(const std::string& state_path) { return impl_->do_recover(state_path); }
Result<void> Scheduler::persist(const std::string& state_path) const { return impl_->do_persist(state_path); }
void Scheduler::roll_epoch() { impl_->do_roll_epoch(); impl_->flush_events(); }

SchedulerConfig default_scheduler_config() {
  SchedulerConfig c;
  c.admission.max_global_admitted = 1024;
  c.admission.max_global_queued = 4096;
  c.admission.max_tenants = 64;
  c.admission.default_tenant_concurrency = 64;
  c.admission.default_tenant_queue_depth = 512;
  c.admission.max_global_token_budget = 0;
  c.admission.enforce_deadline_feasibility = true;
  c.fairness.default_weight = 1.0;
  c.fairness.quantum = 1;
  c.batch.max_batch_size = 16;
  c.batch.max_tokens = 4096;
  c.batch.max_wait = std::chrono::milliseconds(8);
  c.retry.max_attempts = 3;
  c.retry.base_backoff = std::chrono::milliseconds(20);
  c.retry.backoff_multiplier = 2.0;
  c.retry.max_backoff = std::chrono::milliseconds(2000);
  return c;
}

}  // namespace inference_scheduler