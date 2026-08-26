#include <inference_scheduler/inference_scheduler.hpp>
#include <cstdio>
#include <string>

using namespace inference_scheduler;

static int g_fail = 0;
#define CHECK(cond, msg) do { if (!(cond)) { std::printf("FAIL: %s\n", msg); ++g_fail; } } while (0)

static CompletionReport make_report(Scheduler& sched, const WorkerRegistration& wr,
                                    RequestId rid, AttemptId aid) {
  auto snap = sched.request_snapshot(rid);
  CompletionReport r;
  r.epoch = sched.epoch();
  r.worker = wr.worker;
  r.boot_id = wr.boot_id;
  r.request_id = rid;
  r.attempt_id = aid;
  r.generation = snap.ok() ? snap.value().generation : Generation();
  r.worker_snapshot.available_capacity_units = 4;
  r.worker_snapshot.total_capacity_units = 4;
  return r;
}

int main() {
  auto cfg = default_scheduler_config();
  auto clock = std::make_shared<SimulatedClock>();
  Scheduler sched(cfg, clock);

  WorkerRegistration wr;
  wr.worker = WorkerId(1);
  wr.node = NodeId(1);
  wr.accelerator = AcceleratorId(1);
  wr.boot_id = WorkerBootId(1001);
  wr.state = WorkerState::Online;
  wr.capability.backend = "cpu";
  wr.capability.device_name = "cpu0";
  wr.capability.total_capacity_units = 4;
  wr.capability.available_capacity_units = 4;
  sched.register_worker(wr);

  // Batch request with a deadline
  RequestSpec spec;
  spec.tenant = TenantId(2);
  spec.model = ModelIdentity(10);
  spec.revision = ModelRevision(1);
  spec.phase = RequestPhase::Prefill;
  spec.priority = PriorityClass::High;
  spec.latency_class = LatencyClass::Interactive;
  spec.tokens.input = 100;
  spec.tokens.output = 50;
  spec.demand.cost_units = 1.0;
  spec.deadline = Deadline::after(std::chrono::milliseconds(5000), clock->now());

  auto ad = sched.submit(spec);
  CHECK(ad.ok() && ad.value().decision == AdmissionDecision::Admit, "submit admits");
  CHECK(ad.ok() && ad.value().request_id.valid(), "request id assigned");

  auto d = sched.plan_dispatch(ad.value().request_id, true);
  CHECK(d.ok(), "plan_dispatch ok");
  CHECK(d.ok() && d.value().decision == DispatchDecision::Dispatch, "dispatch decision dispatch");
  CHECK(d.ok() && d.value().worker_id.valid(), "worker selected");
  CHECK(d.ok() && !d.value().batch_members.empty(), "batch members non-empty");

  // confirm every member start then complete
  for (const auto rid : d.value().batch_members) {
    auto snap = sched.request_snapshot(rid);
    CHECK(snap.ok() && snap.value().generation.valid(), "generation valid");
    auto start = make_report(sched, wr, rid, snap.value().attempt_id);
    auto cs = sched.confirm_dispatch_start(start);
    CHECK(cs.ok(), "confirm start ok");
    auto comp = make_report(sched, wr, rid, snap.value().attempt_id);
    comp.status = CompletionStatus::Succeeded;
    comp.worker_snapshot.available_capacity_units = 4;
    auto co = sched.complete_attempt(comp);
    CHECK(co.ok() && co.value().acceptance == CompletionAcceptance::Accepted, "complete accepted");
  }

  auto st = sched.stats();
  CHECK(st.requests_admitted == 1, "admitted stat");
  CHECK(st.requests_completed == 1, "completed stat");
  CHECK(st.current_queued == 0 && st.current_running == 0, "no leaked queue/running");

  sched.roll_epoch();
  // A completion from the old epoch must be rejected as stale
  auto snap = sched.request_snapshot(ad.value().request_id);
  CompletionReport stale;
  stale.epoch = SchedulerEpoch(1);
  stale.worker = wr.worker;
  stale.boot_id = wr.boot_id;
  stale.request_id = ad.value().request_id;
  stale.attempt_id = snap.value().attempt_id;
  stale.generation = snap.value().generation;
  stale.status = CompletionStatus::Succeeded;
  auto co2 = sched.complete_attempt(stale);
  CHECK(co2.ok() && co2.value().acceptance == CompletionAcceptance::StaleEpoch, "stale epoch rejected");

  if (g_fail == 0) { std::printf("SMOKE PASS\n"); return 0; }
  std::printf("SMOKE FAIL (%d)\n", g_fail);
  return 1;
}
