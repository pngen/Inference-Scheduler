// Scheduler invariants: admission accounting, attempt authority, capacity,
// batching, worker loss, epoch rollover and recovery. Each scenario stands
// alone; naming one on the command line runs only that one.
#include "test_util.hpp"
#include "json_util.hpp"
#include "persistence.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

using namespace ts;

namespace {

std::shared_ptr<SimulatedClock> sim_clock() { return std::make_shared<SimulatedClock>(); }

// A worker with boot id 10 * id that serves only the listed models (any model
// when the list is empty).
WorkerRegistration worker_for(std::uint64_t id, int units, const std::vector<std::uint64_t>& models = {}) {
  WorkerRegistration w = make_worker(WorkerId(id), WorkerBootId(id * 10), units);
  for (const std::uint64_t m : models) w.capability.models.push_back(ModelIdentity(m));
  return w;
}

RequestSpec spec_for(std::uint64_t tenant, std::uint64_t model, double cost = 1.0,
                     RequestPhase phase = RequestPhase::Prefill) {
  return make_spec(TenantId(tenant), ModelIdentity(model), phase, cost);
}

// A report from worker `id` (boot id 10 * id) about the request's current
// attempt. `available`/`total` are the capacity the worker reports with it; the
// defaults report nothing.
CompletionReport report_from(Scheduler& s, std::uint64_t id, RequestId r, int available = -1, int total = 0) {
  auto snap = s.request_snapshot(r);
  CompletionReport c;
  c.epoch = s.epoch();
  c.worker = WorkerId(id);
  c.boot_id = WorkerBootId(id * 10);
  c.request_id = r;
  if (snap.ok()) { c.attempt_id = snap.value().attempt_id; c.generation = snap.value().generation; }
  c.worker_snapshot.available_capacity_units = available;
  c.worker_snapshot.total_capacity_units = total;
  return c;
}

RequestState state_of(Scheduler& s, RequestId r) {
  auto snap = s.request_snapshot(r);
  return snap.ok() ? snap.value().state : RequestState::Created;
}

bool dispatched(const Result<DispatchOutcome>& d) {
  return d.ok() && d.value().decision == DispatchDecision::Dispatch;
}

void s_duplicate_request_id() {
  Scheduler s(default_scheduler_config(), sim_clock());
  s.register_worker(worker_for(1, 8));
  const RequestSpec spec = spec_for(1, 10);
  REQUIRE(s.submit(spec, RequestId(1)).ok(), "first submit");
  REQUIRE(dispatched(s.plan_dispatch(RequestId(1))), "reserved on the worker");
  auto again = s.submit(spec, RequestId(1));
  REQUIRE(!again.ok() && again.error().code == ErrorCode::AlreadyExists, "an id in use is refused");
  REQUIRE(state_of(s, RequestId(1)) == RequestState::Reserved, "the request holding the id is untouched");
  REQUIRE(s.queued_count() == 0, "nothing was queued a second time");
  REQUIRE(s.stats().current_admitted == 1, "admitted once");
  REQUIRE(!s.submit(spec, RequestId()).ok(), "an invalid preassigned id is refused");
}

void s_token_budget() {
  auto cfg = default_scheduler_config();
  cfg.admission.max_global_token_budget = 100;  // each request below is 32 tokens
  Scheduler s(cfg, sim_clock());
  s.register_worker(worker_for(1, 8));
  for (std::uint64_t i = 1; i <= 3; ++i) {
    auto ad = s.submit(spec_for(1, 10), RequestId(i));
    REQUIRE(ad.ok() && ad.value().decision == AdmissionDecision::Admit, "within the budget");
  }
  auto over = s.submit(spec_for(1, 10), RequestId(4));
  REQUIRE(over.ok() && over.value().decision == AdmissionDecision::Defer, "the budget is cumulative");
  REQUIRE(over.ok() && over.value().reason_code == "token_budget_exhausted", "budget reason");
  REQUIRE_OK(s.cancel(RequestId(1), CancellationReason::ClientRequested, "free the tokens"));
  auto after = s.submit(spec_for(1, 10), RequestId(5));
  REQUIRE(after.ok() && after.value().decision == AdmissionDecision::Admit, "tokens return when a request ends");
}

void s_tenant_limit() {
  auto cfg = default_scheduler_config();
  cfg.admission.max_tenants = 2;
  Scheduler s(cfg, sim_clock());
  s.register_worker(worker_for(1, 8));
  REQUIRE(s.submit(spec_for(1, 10), RequestId(1)).value().decision == AdmissionDecision::Admit, "tenant 1");
  REQUIRE(s.submit(spec_for(2, 10), RequestId(2)).value().decision == AdmissionDecision::Admit, "tenant 2");
  auto third = s.submit(spec_for(3, 10), RequestId(3));
  REQUIRE(third.ok() && third.value().decision == AdmissionDecision::Defer, "a third active tenant is deferred");
  REQUIRE(third.ok() && third.value().reason_code == "tenant_limit_reached", "tenant limit reason");
  REQUIRE(s.submit(spec_for(1, 10), RequestId(4)).value().decision == AdmissionDecision::Admit, "an active tenant is not limited");
  REQUIRE_OK(s.cancel(RequestId(2), CancellationReason::ClientRequested, "tenant 2 goes idle"));
  REQUIRE(s.submit(spec_for(3, 10), RequestId(5)).value().decision == AdmissionDecision::Admit, "room once a tenant goes idle");
}

void s_class_values() {
  Scheduler s(default_scheduler_config(), sim_clock());
  s.register_worker(worker_for(1, 8));
  RequestSpec latency = spec_for(1, 10);
  latency.latency_class = static_cast<LatencyClass>(9);
  REQUIRE(!s.submit(latency).ok(), "latency class out of range");
  RequestSpec priority = spec_for(1, 10);
  priority.priority = static_cast<PriorityClass>(77);
  REQUIRE(!s.submit(priority).ok(), "priority out of range");
  RequestSpec phase = spec_for(1, 10);
  phase.phase = static_cast<RequestPhase>(-1);
  REQUIRE(!s.submit(phase).ok(), "phase out of range");
  REQUIRE(s.stats().requests_admitted == 0, "none of them was admitted");
}

void s_numeric_bounds() {
  Scheduler s(default_scheduler_config(), sim_clock());
  s.register_worker(worker_for(1, 8));
  RequestSpec nan_cost = spec_for(1, 10);
  nan_cost.demand.cost_units = std::nan("");
  REQUIRE(!s.submit(nan_cost).ok(), "a cost that is not a number");
  RequestSpec inf_cost = spec_for(1, 10);
  inf_cost.demand.cost_units = std::numeric_limits<double>::infinity();
  REQUIRE(!s.submit(inf_cost).ok(), "an infinite cost");
  RequestSpec huge_cost = spec_for(1, 10);
  huge_cost.demand.cost_units = 3.0e9;  // beyond any worker's int capacity
  REQUIRE(!s.submit(huge_cost).ok(), "a cost no worker can hold");
  RequestSpec tokens = spec_for(1, 10);
  tokens.tokens.input = std::numeric_limits<std::int64_t>::max();
  tokens.tokens.output = 1;
  REQUIRE(!s.submit(tokens).ok(), "a token total that overflows");
  REQUIRE(s.submit(spec_for(1, 10)).ok(), "an ordinary request is still accepted");
}

void s_batch_fits_capacity() {
  Scheduler s(default_scheduler_config(), sim_clock());
  s.register_worker(worker_for(1, 4));
  for (std::uint64_t i = 1; i <= 6; ++i) s.submit(spec_for(1, 10), RequestId(i));
  auto d = s.plan_next_dispatch();
  REQUIRE(dispatched(d), "a batch that fits the worker is dispatched");
  REQUIRE(d.ok() && d.value().batch_members.size() == 4, "the batch is cut to the worker's free units");
  REQUIRE(s.queued_count() == 2, "the rest stays queued");

  // Fractional costs: the units asked of a worker are the units reserved on it.
  Scheduler f(default_scheduler_config(), sim_clock());
  f.register_worker(worker_for(1, 5));
  for (std::uint64_t i = 1; i <= 3; ++i) f.submit(spec_for(1, 10, 1.5), RequestId(i));
  auto d2 = f.plan_dispatch(RequestId(1));
  REQUIRE(dispatched(d2), "fractional costs dispatch");
  REQUIRE(d2.ok() && d2.value().batch_members.size() == 2, "two members of two units each fit five");
  auto ws = f.worker_snapshot(WorkerId(1));
  REQUIRE(ws.ok() && ws.value().reserved_units == 4, "reserved what was asked for");
}

void s_member_batch_cap() {
  Scheduler s(default_scheduler_config(), sim_clock());
  s.register_worker(worker_for(1, 100));
  s.submit(spec_for(1, 10), RequestId(1));
  RequestSpec alone = spec_for(1, 10);
  alone.max_batch_size = 1;  // must run by itself
  s.submit(alone, RequestId(2));
  s.submit(spec_for(1, 10), RequestId(3));
  auto d = s.plan_dispatch(RequestId(1));
  REQUIRE(dispatched(d), "dispatch");
  bool pulled_in = false;
  for (const RequestId m : d.value().batch_members) { if (m == RequestId(2)) pulled_in = true; }
  REQUIRE(!pulled_in, "a request capped at one is not pulled into another batch");
  REQUIRE(d.value().batch_members.size() == 2, "the uncapped requests still batch");
  auto d2 = s.plan_dispatch(RequestId(2));
  REQUIRE(dispatched(d2) && d2.value().batch_members.size() == 1, "and it runs alone");
}

void s_cancel_reserved() {
  Scheduler s(default_scheduler_config(), sim_clock());
  s.register_worker(worker_for(1, 8));
  s.submit(spec_for(1, 10, 3.0), RequestId(1));
  REQUIRE(dispatched(s.plan_dispatch(RequestId(1))), "reserved");
  REQUIRE(s.worker_snapshot(WorkerId(1)).value().reserved_units == 3, "three units reserved");
  REQUIRE_OK(s.cancel(RequestId(1), CancellationReason::ClientRequested, "before start"));
  REQUIRE(state_of(s, RequestId(1)) == RequestState::Cancelled, "cancelled");
  REQUIRE(s.worker_snapshot(WorkerId(1)).value().reserved_units == 0, "the reservation is released");
  REQUIRE(s.available_worker_capacity_units() == 8, "capacity is back to baseline");
  REQUIRE(s.stats().current_reserved_units == 0, "nothing reserved globally");
}

void s_assigned_worker_authority() {
  Scheduler s(default_scheduler_config(), sim_clock());
  s.register_worker(worker_for(1, 8, {10}));
  s.register_worker(worker_for(2, 8, {20}));
  s.submit(spec_for(1, 10), RequestId(1));
  auto d = s.plan_dispatch(RequestId(1));
  REQUIRE(dispatched(d) && d.value().worker_id == WorkerId(1), "reserved on worker 1");
  // Worker 2 is registered and alive, but the attempt is not its to report on.
  CompletionReport wrong = report_from(s, 2, RequestId(1));
  auto cs = s.confirm_dispatch_start(wrong);
  REQUIRE(!cs.ok() && cs.error().code == ErrorCode::StaleWorker, "another worker cannot start the attempt");
  wrong.status = CompletionStatus::Succeeded;
  auto co = s.complete_attempt(wrong);
  REQUIRE(co.ok() && co.value().acceptance == CompletionAcceptance::StaleWorker, "another worker cannot complete it");
  REQUIRE(state_of(s, RequestId(1)) == RequestState::Reserved, "the attempt is still reserved");
  REQUIRE_OK(s.confirm_dispatch_start(report_from(s, 1, RequestId(1))));
  CompletionReport done = report_from(s, 1, RequestId(1));
  auto ok = s.complete_attempt(done);
  REQUIRE(ok.ok() && ok.value().acceptance == CompletionAcceptance::Accepted, "the assigned worker completes it");
  auto ws = s.worker_snapshot(WorkerId(1));
  REQUIRE(ws.ok() && ws.value().reserved_units == 0 && ws.value().running == 0, "worker 1 is back to baseline");
  REQUIRE(s.stats().stale_rejected >= 1, "the foreign completion was counted as stale");
}

void s_completion_needs_dispatch() {
  Scheduler s(default_scheduler_config(), sim_clock());
  s.register_worker(worker_for(1, 8));
  auto ad = s.submit(spec_for(1, 10), RequestId(1));
  REQUIRE(ad.ok() && ad.value().decision == AdmissionDecision::Admit, "admitted");
  // The attempt identity is known from admission, but nothing was dispatched.
  CompletionReport c = report_from(s, 1, RequestId(1));
  c.status = CompletionStatus::Succeeded;
  auto co = s.complete_attempt(c);
  REQUIRE(co.ok() && co.value().acceptance == CompletionAcceptance::Invalid, "a queued attempt cannot be completed");
  REQUIRE(state_of(s, RequestId(1)) == RequestState::Queued, "still queued");
  REQUIRE(s.stats().requests_completed == 0, "nothing completed");
  s.submit(spec_for(1, 10), RequestId(2));
  REQUIRE(dispatched(s.plan_next_dispatch()), "the queue still moves");
}

void s_stale_report() {
  Scheduler s(default_scheduler_config(), sim_clock());
  s.register_worker(worker_for(1, 8));
  s.submit(spec_for(1, 10), RequestId(1));
  REQUIRE(dispatched(s.plan_dispatch(RequestId(1))), "reserved");
  REQUIRE_OK(s.confirm_dispatch_start(report_from(s, 1, RequestId(1))));
  const AttemptId before = s.request_snapshot(RequestId(1)).value().attempt_id;
  CompletionReport stale = report_from(s, 1, RequestId(1));
  stale.status = CompletionStatus::Stale;
  auto co = s.complete_attempt(stale);
  REQUIRE(co.ok() && co.value().acceptance == CompletionAcceptance::StaleAttempt, "reported stale");
  REQUIRE(state_of(s, RequestId(1)) == RequestState::Queued, "the request is not left running without an attempt");
  REQUIRE(s.request_snapshot(RequestId(1)).value().attempt_id != before, "it has a new attempt");
  auto ws = s.worker_snapshot(WorkerId(1));
  REQUIRE(ws.ok() && ws.value().reserved_units == 0 && ws.value().running == 0, "worker released");
}

void s_retry_flags() {
  auto run_failure = [](SchedulerConfig cfg, FailureClass cls, bool& retried, RequestState& state) {
    Scheduler s(cfg, sim_clock());
    s.register_worker(worker_for(1, 8));
    s.submit(spec_for(1, 10), RequestId(1));
    (void)s.plan_dispatch(RequestId(1));
    (void)s.confirm_dispatch_start(report_from(s, 1, RequestId(1)));
    CompletionReport fail = report_from(s, 1, RequestId(1));
    fail.status = CompletionStatus::Failed;
    fail.failure_class = cls;
    auto co = s.complete_attempt(fail);
    retried = co.ok() && co.value().retried;
    state = state_of(s, RequestId(1));
  };
  bool retried = false;
  RequestState state = RequestState::Created;
  run_failure(default_scheduler_config(), FailureClass::RetryableWorker, retried, state);
  REQUIRE(retried && state == RequestState::Queued, "worker failures retry by default");
  auto no_worker = default_scheduler_config();
  no_worker.retry.retry_on_worker_failure = false;
  run_failure(no_worker, FailureClass::RetryableWorker, retried, state);
  REQUIRE(!retried && state == RequestState::Failed, "retry_on_worker_failure = false is honoured");
  auto no_transport = default_scheduler_config();
  no_transport.retry.retry_on_transport = false;
  run_failure(no_transport, FailureClass::RetryableTransport, retried, state);
  REQUIRE(!retried && state == RequestState::Failed, "retry_on_transport = false is honoured");
}

void s_failure_after_cancel() {
  Scheduler s(default_scheduler_config(), sim_clock());
  s.register_worker(worker_for(1, 8));
  s.submit(spec_for(1, 10), RequestId(1));
  REQUIRE(dispatched(s.plan_dispatch(RequestId(1))), "reserved");
  REQUIRE_OK(s.confirm_dispatch_start(report_from(s, 1, RequestId(1))));
  REQUIRE_OK(s.cancel(RequestId(1), CancellationReason::ClientRequested, "while running"));
  CompletionReport fail = report_from(s, 1, RequestId(1));
  fail.status = CompletionStatus::Failed;
  fail.failure_class = FailureClass::RetryableWorker;
  auto co = s.complete_attempt(fail);
  REQUIRE(co.ok() && co.value().acceptance == CompletionAcceptance::CancelledOutcome, "the cancellation commits");
  REQUIRE(state_of(s, RequestId(1)) == RequestState::Cancelled, "the request ends cancelled, not failed");
  const auto st = s.stats();
  REQUIRE(st.requests_cancelled == 1 && st.requests_failed == 0, "counted as a cancellation");
}

void s_worker_loss() {
  // A worker that goes away with work it has not started.
  Scheduler s(default_scheduler_config(), sim_clock());
  s.register_worker(worker_for(1, 8));
  s.submit(spec_for(1, 10), RequestId(1));
  s.submit(spec_for(1, 10), RequestId(2));
  auto d = s.plan_dispatch(RequestId(1));
  REQUIRE(dispatched(d) && d.value().batch_members.size() == 2, "both reserved in one batch");
  const AttemptId a1 = s.request_snapshot(RequestId(1)).value().attempt_id;
  const AttemptId a2 = s.request_snapshot(RequestId(2)).value().attempt_id;
  REQUIRE_OK(s.unregister_worker(WorkerId(1)));
  REQUIRE(state_of(s, RequestId(1)) == RequestState::Queued && state_of(s, RequestId(2)) == RequestState::Queued, "reserved work goes back to the queue");
  REQUIRE(s.request_snapshot(RequestId(1)).value().attempt_id != a1 && s.request_snapshot(RequestId(2)).value().attempt_id != a2, "under new attempts");
  auto st = s.stats();
  REQUIRE(st.current_reserved == 0 && st.current_reserved_units == 0 && st.current_capacity_units == 0, "no reservation survives the worker");
  s.register_worker(worker_for(2, 8));
  auto d2 = s.plan_next_dispatch();
  REQUIRE(dispatched(d2) && d2.value().worker_id == WorkerId(2), "and runs elsewhere");

  // A worker that restarts with one member of its batch started.
  Scheduler r(default_scheduler_config(), sim_clock());
  r.register_worker(worker_for(1, 8));
  r.submit(spec_for(1, 10), RequestId(1));
  r.submit(spec_for(1, 10), RequestId(2));
  REQUIRE(dispatched(r.plan_dispatch(RequestId(1))), "both reserved");
  REQUIRE_OK(r.confirm_dispatch_start(report_from(r, 1, RequestId(1))));
  CompletionReport old_boot = report_from(r, 1, RequestId(1));
  old_boot.status = CompletionStatus::Succeeded;
  r.register_worker(make_worker(WorkerId(1), WorkerBootId(999), 8));
  REQUIRE(state_of(r, RequestId(1)) == RequestState::Queued && state_of(r, RequestId(2)) == RequestState::Queued, "started and unstarted members are both recovered");
  auto ws = r.worker_snapshot(WorkerId(1));
  REQUIRE(ws.ok() && ws.value().reserved_units == 0 && ws.value().running == 0, "the restarted worker holds nothing");
  st = r.stats();
  REQUIRE(st.current_capacity_units == 0 && st.current_reserved_units == 0, "global units are back to zero");
  auto late = r.complete_attempt(old_boot);
  REQUIRE(late.ok() && late.value().acceptance == CompletionAcceptance::StaleWorker, "the old boot's report is stale");
}

void s_epoch_rollover() {
  Scheduler s(default_scheduler_config(), sim_clock());
  s.register_worker(worker_for(1, 8));
  s.submit(spec_for(1, 10), RequestId(1));
  REQUIRE(dispatched(s.plan_dispatch(RequestId(1))), "reserved");
  REQUIRE_OK(s.confirm_dispatch_start(report_from(s, 1, RequestId(1))));
  CompletionReport old_report = report_from(s, 1, RequestId(1));
  old_report.status = CompletionStatus::Succeeded;
  s.roll_epoch();
  REQUIRE(state_of(s, RequestId(1)) == RequestState::Queued, "work issued under the old epoch is recovered");
  auto ws = s.worker_snapshot(WorkerId(1));
  REQUIRE(ws.ok() && ws.value().reserved_units == 0 && ws.value().running == 0, "the worker holds nothing from the old epoch");
  auto stale_epoch = s.complete_attempt(old_report);
  REQUIRE(stale_epoch.ok() && stale_epoch.value().acceptance == CompletionAcceptance::StaleEpoch, "old epoch rejected");
  old_report.epoch = s.epoch();
  auto stale_attempt = s.complete_attempt(old_report);
  REQUIRE(stale_attempt.ok() && stale_attempt.value().acceptance == CompletionAcceptance::StaleAttempt, "old attempt rejected under the new epoch");
  REQUIRE(dispatched(s.plan_dispatch(RequestId(1))), "and it is dispatched again under the new epoch");
}

void s_running_counts_against_capacity() {
  Scheduler s(default_scheduler_config(), sim_clock());
  s.register_worker(worker_for(1, 4));
  s.submit(spec_for(1, 10, 4.0), RequestId(1));
  s.submit(spec_for(1, 11, 4.0), RequestId(2));  // another model: its own batch
  REQUIRE(dispatched(s.plan_dispatch(RequestId(1))), "first fills the worker");
  // The worker reports its capacity the way every caller here does: the total.
  REQUIRE_OK(s.confirm_dispatch_start(report_from(s, 1, RequestId(1), 4, 4)));
  auto second = s.plan_dispatch(RequestId(2));
  REQUIRE(second.ok() && second.value().decision != DispatchDecision::Dispatch, "a full worker is not handed more while it runs");
  CompletionReport done = report_from(s, 1, RequestId(1), 4, 4);
  REQUIRE(s.complete_attempt(done).value().acceptance == CompletionAcceptance::Accepted, "first completes");
  REQUIRE(dispatched(s.plan_dispatch(RequestId(2))), "then the second is dispatched");
}

void s_capacity_reports() {
  Scheduler s(default_scheduler_config(), sim_clock());
  s.register_worker(worker_for(1, 8));
  s.submit(spec_for(1, 10, 2.0), RequestId(1));
  REQUIRE(dispatched(s.plan_dispatch(RequestId(1))), "reserved");
  REQUIRE_OK(s.confirm_dispatch_start(report_from(s, 1, RequestId(1), 6, 8)));
  REQUIRE(s.worker_snapshot(WorkerId(1)).value().available_capacity_units == 6, "free units as reported at start");
  CompletionReport done = report_from(s, 1, RequestId(1), 8, 8);
  REQUIRE(s.complete_attempt(done).value().acceptance == CompletionAcceptance::Accepted, "completed");
  REQUIRE(s.worker_snapshot(WorkerId(1)).value().available_capacity_units == 8, "free units as reported at completion");
  REQUIRE(s.available_worker_capacity_units() == 8, "capacity is back to baseline");

  // A report cannot claim more than the worker registered with.
  s.submit(spec_for(1, 10, 2.0), RequestId(2));
  REQUIRE(dispatched(s.plan_dispatch(RequestId(2))), "reserved");
  REQUIRE_OK(s.confirm_dispatch_start(report_from(s, 1, RequestId(2), 100, 100)));
  REQUIRE(s.worker_snapshot(WorkerId(1)).value().available_capacity_units <= 8, "capped at the registered capacity");
  REQUIRE(s.available_worker_capacity_units() <= 8, "no capacity appears from a report");
}

void s_plan_next_blocked_head() {
  Scheduler s(default_scheduler_config(), sim_clock());
  s.register_worker(worker_for(1, 8, {10}));
  s.register_worker(worker_for(2, 8, {20}));
  REQUIRE_OK(s.set_worker_state(WorkerId(1), WorkerState::Offline));
  s.submit(spec_for(1, 10), RequestId(1));  // heads the fair order, but its only worker is offline
  s.submit(spec_for(2, 20), RequestId(2));
  auto d = s.plan_next_dispatch();
  REQUIRE(dispatched(d), "a head that cannot be placed does not block the next flow");
  REQUIRE(d.ok() && d.value().request_id == RequestId(2) && d.value().worker_id == WorkerId(2), "tenant 2 runs on its worker");
  REQUIRE(state_of(s, RequestId(1)) == RequestState::Queued, "the blocked request keeps waiting");
}

void s_phase_mix_member() {
  auto cfg = default_scheduler_config();
  cfg.batch.allow_phase_mix = true;
  Scheduler s(cfg, sim_clock());
  WorkerRegistration prefill_only = worker_for(1, 8);
  prefill_only.capability.phases.push_back(RequestPhase::Prefill);
  WorkerRegistration decode_only = worker_for(2, 8);
  decode_only.capability.phases.push_back(RequestPhase::Decode);
  s.register_worker(prefill_only);
  s.register_worker(decode_only);
  REQUIRE_OK(s.set_worker_state(WorkerId(2), WorkerState::Offline));
  s.submit(spec_for(1, 10, 1.0, RequestPhase::Prefill), RequestId(1));
  s.submit(spec_for(1, 10, 1.0, RequestPhase::Decode), RequestId(2));
  auto d = s.plan_dispatch(RequestId(1));
  REQUIRE(dispatched(d) && d.value().worker_id == WorkerId(1), "the prefill request is dispatched");
  REQUIRE(d.ok() && d.value().batch_members.size() == 1, "without a member its worker cannot run");
  REQUIRE(state_of(s, RequestId(2)) == RequestState::Queued, "the decode request waits for a worker that runs decode");
}

void s_explain_admitted() {
  auto cfg = default_scheduler_config();
  cfg.admission.max_global_admitted = 1;
  Scheduler s(cfg, sim_clock());
  s.register_worker(worker_for(1, 8));
  REQUIRE(s.submit(spec_for(1, 10), RequestId(1)).value().decision == AdmissionDecision::Admit, "admitted");
  auto ex = s.explain(RequestId(1));
  REQUIRE(ex.ok() && ex.value().admitted && ex.value().admission == AdmissionDecision::Admit, "an admitted request is explained as admitted");
}

void s_expired_by_class() {
  auto clock = sim_clock();
  Scheduler s(default_scheduler_config(), clock);
  s.register_worker(worker_for(1, 8));
  RequestSpec spec = spec_for(1, 10);
  spec.latency_class = LatencyClass::Interactive;
  spec.deadline = Deadline::after(std::chrono::milliseconds(40), clock->now());
  REQUIRE(s.submit(spec, RequestId(1)).value().decision == AdmissionDecision::Admit, "admitted");
  clock->advance(std::chrono::milliseconds(100));
  (void)s.plan_dispatch(RequestId(1));
  REQUIRE(state_of(s, RequestId(1)) == RequestState::Expired, "expired");
  const auto st = s.stats();
  REQUIRE(st.by_latency[latency_index(LatencyClass::Interactive)].expired == 1, "counted under its latency class");
}

void s_dispatches_counted_once() {
  Scheduler s(default_scheduler_config(), sim_clock());
  s.register_worker(worker_for(1, 8));
  s.submit(spec_for(1, 10), RequestId(1));
  REQUIRE(dispatched(s.plan_dispatch(RequestId(1))), "reserved");
  REQUIRE_OK(s.confirm_dispatch_start(report_from(s, 1, RequestId(1))));
  CompletionReport done = report_from(s, 1, RequestId(1));
  REQUIRE(s.complete_attempt(done).value().acceptance == CompletionAcceptance::Accepted, "completed");
  REQUIRE(s.stats().dispatches == 1, "one dispatch is counted once");
}

void s_recover_bad_payload() {
  auto cfg = default_scheduler_config();
  cfg.persist_recovery = true;
  Scheduler s(cfg, sim_clock());
  s.register_worker(worker_for(1, 8));
  for (std::uint64_t i = 1; i <= 3; ++i) s.submit(spec_for(1, 10), RequestId(i));
  // A well-formed envelope around a payload that ends right after its header.
  std::string payload;
  const std::uint32_t magic = 0x49534348u, version = 1u;
  payload.append(reinterpret_cast<const char*>(&magic), 4);
  payload.append(reinterpret_cast<const char*>(&version), 4);
  const std::string path = "is_invariant_bad.bin";
  REQUIRE_OK(inference_scheduler::detail::write_persistence_file(path, payload));
  auto rc = s.recover(path);
  REQUIRE(!rc.ok() && rc.error().code == ErrorCode::Corruption, "the payload is refused");
  REQUIRE(s.queued_count() == 3, "the live queue is untouched");
  REQUIRE(state_of(s, RequestId(2)) == RequestState::Queued, "the live requests are untouched");
  REQUIRE(s.stats().current_admitted == 3, "the live counters are untouched");
  REQUIRE(dispatched(s.plan_next_dispatch()), "and the scheduler still schedules");
  std::remove(path.c_str());
  std::remove((path + ".tmp").c_str());
}

void s_recover_requeues_and_rebuilds() {
  auto cfg = default_scheduler_config();
  cfg.persist_recovery = true;
  cfg.batch.max_batch_size = 1;
  Scheduler s(cfg, sim_clock());
  s.register_worker(worker_for(1, 8));
  for (std::uint64_t i = 1; i <= 3; ++i) s.submit(spec_for(1, 10), RequestId(i));
  REQUIRE(dispatched(s.plan_dispatch(RequestId(1))), "request 1 reserved");
  REQUIRE_OK(s.confirm_dispatch_start(report_from(s, 1, RequestId(1))));
  REQUIRE(dispatched(s.plan_dispatch(RequestId(2))), "request 2 reserved");
  CompletionReport late = report_from(s, 1, RequestId(1));
  late.status = CompletionStatus::Succeeded;
  const AttemptId a2 = s.request_snapshot(RequestId(2)).value().attempt_id;
  const AttemptId a3 = s.request_snapshot(RequestId(3)).value().attempt_id;
  const std::string path = "is_invariant_state.bin";
  REQUIRE_OK(s.persist(path));

  Scheduler s2(cfg, sim_clock());
  REQUIRE_OK(s2.recover(path));
  for (std::uint64_t i = 1; i <= 3; ++i) {
    REQUIRE(state_of(s2, RequestId(i)) == RequestState::Queued, "every unfinished request is queued again");
  }
  REQUIRE(s2.request_snapshot(RequestId(1)).value().attempt_id != late.attempt_id, "running work comes back as a new attempt");
  REQUIRE(s2.request_snapshot(RequestId(2)).value().attempt_id != a2, "reserved work comes back as a new attempt");
  REQUIRE(s2.request_snapshot(RequestId(3)).value().attempt_id == a3, "queued work keeps its attempt");
  const auto st = s2.stats();
  REQUIRE(st.current_admitted == 3 && st.current_queued == 3, "admission counters match the requests");
  REQUIRE(st.current_running == 0 && st.current_reserved == 0 && st.current_capacity_units == 0, "nothing is held");
  late.epoch = s2.epoch();
  auto co = s2.complete_attempt(late);
  REQUIRE(co.ok() && co.value().acceptance != CompletionAcceptance::Accepted, "a late report for the old attempt is not accepted");
  REQUIRE(state_of(s2, RequestId(1)) == RequestState::Queued, "and does not complete the request");

  auto original = inference_scheduler::detail::read_persistence_file(path);
  REQUIRE(original.ok(), "payload readable");
  const std::string edited_path = "is_invariant_edit.bin";
  {
    // Stored counters are not trusted: here the admitted count claims a million.
    std::string p = original.value();
    const std::int64_t million = 1000000;
    std::memcpy(&p[16], &million, sizeof(million));
    REQUIRE_OK(inference_scheduler::detail::write_persistence_file(edited_path, p));
    Scheduler s3(cfg, sim_clock());
    REQUIRE_OK(s3.recover(edited_path));
    auto ad = s3.submit(spec_for(2, 10), RequestId(50));
    REQUIRE(ad.ok() && ad.value().decision == AdmissionDecision::Admit, "admission is judged by the requests that exist");
  }
  {
    std::string p = original.value() + "junk";
    REQUIRE_OK(inference_scheduler::detail::write_persistence_file(edited_path, p));
    Scheduler s4(cfg, sim_clock());
    REQUIRE(!s4.recover(edited_path).ok(), "trailing bytes are refused");
  }
  {
    // The first request record's state, set to a value no state has.
    std::string p = original.value();
    const std::int32_t impossible = 99;
    std::memcpy(&p[144], &impossible, sizeof(impossible));
    REQUIRE_OK(inference_scheduler::detail::write_persistence_file(edited_path, p));
    Scheduler s5(cfg, sim_clock());
    REQUIRE(!s5.recover(edited_path).ok(), "an impossible enum value is refused");
  }
  std::remove(path.c_str());
  std::remove((path + ".tmp").c_str());
  std::remove(edited_path.c_str());
  std::remove((edited_path + ".tmp").c_str());
}

void s_local_driver_worker_units() {
  Scheduler s(default_scheduler_config(), sim_clock());
  const WorkerRegistration small = worker_for(1, 4, {10});
  const WorkerRegistration large = worker_for(2, 64, {20});
  s.register_worker(small);
  s.register_worker(large);
  LocalDriver drv(s, std::make_shared<CpuExecutor>(10, 1000), {small, large});
  s.submit(spec_for(1, 20), RequestId(1));
  drv.run_until_idle();
  REQUIRE(state_of(s, RequestId(1)) == RequestState::Completed, "ran on the large worker");
  REQUIRE(s.worker_snapshot(WorkerId(2)).value().available_capacity_units == 64, "the large worker keeps its own capacity");
}

void s_json_depth() {
  bool ok = true;
  (void)net::parse_json(std::string(200000, '['), ok);
  REQUIRE(!ok, "deeply nested arrays are refused");
  ok = true;
  std::string objects;
  for (int i = 0; i < 50000; ++i) objects += "{\"a\":";
  (void)net::parse_json(objects, ok);
  REQUIRE(!ok, "deeply nested objects are refused");
  ok = true;
  const net::Json msg = net::parse_json("{\"op\":\"run\",\"members\":[{\"req\":\"1\"}]}", ok);
  REQUIRE(ok && msg.is_obj(), "a protocol message still parses");
}

struct Scenario { const char* name; void (*run)(); };
const Scenario k_scenarios[] = {
    {"duplicate_request_id", s_duplicate_request_id},
    {"token_budget", s_token_budget},
    {"tenant_limit", s_tenant_limit},
    {"class_values", s_class_values},
    {"numeric_bounds", s_numeric_bounds},
    {"batch_fits_capacity", s_batch_fits_capacity},
    {"member_batch_cap", s_member_batch_cap},
    {"cancel_reserved", s_cancel_reserved},
    {"assigned_worker_authority", s_assigned_worker_authority},
    {"completion_needs_dispatch", s_completion_needs_dispatch},
    {"stale_report", s_stale_report},
    {"retry_flags", s_retry_flags},
    {"failure_after_cancel", s_failure_after_cancel},
    {"worker_loss", s_worker_loss},
    {"epoch_rollover", s_epoch_rollover},
    {"running_counts_against_capacity", s_running_counts_against_capacity},
    {"capacity_reports", s_capacity_reports},
    {"plan_next_blocked_head", s_plan_next_blocked_head},
    {"phase_mix_member", s_phase_mix_member},
    {"explain_admitted", s_explain_admitted},
    {"expired_by_class", s_expired_by_class},
    {"dispatches_counted_once", s_dispatches_counted_once},
    {"recover_bad_payload", s_recover_bad_payload},
    {"recover_requeues_and_rebuilds", s_recover_requeues_and_rebuilds},
    {"local_driver_worker_units", s_local_driver_worker_units},
    {"json_depth", s_json_depth},
};

}  // namespace

int main(int argc, char** argv) {
  const std::string only = argc > 1 ? argv[1] : "";
  int ran = 0;
  for (const Scenario& sc : k_scenarios) {
    if (!only.empty() && only != sc.name) continue;
    const int before = g_failures;
    sc.run();
    std::printf("  %s %s\n", g_failures == before ? "ok  " : "FAIL", sc.name);
    ++ran;
  }
  if (ran == 0) { std::printf("no such scenario: %s\n", only.c_str()); return 2; }
  return test_result("invariant_test");
}
