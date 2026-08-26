#include "inference_scheduler/inference_scheduler.hpp"
#include "inference_scheduler/executor.hpp"
#include "inference_scheduler/local_driver.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>

using namespace inference_scheduler;

// Minimal but fully functional CLI over the scheduler runtime. Local commands
// drive an in-process scheduler + CPU worker; serve/worker delegate to the
// distributed coordinator and worker executables. Persistence is optional via
// -state <file>.

static SchedulerConfig cfg_from_args(int argc, char** argv, int& i, std::string& state) {
  SchedulerConfig c = default_scheduler_config();
  bool persist = false;
  for (; i < argc; ++i) { std::string a = argv[i]; if (a == "-state" && i + 1 < argc) { state = argv[++i]; persist = true; } }
  c.persist_recovery = persist;
  c.state_path = state;
  return c;
}

static void print_admission(const AdmissionOutput& o) {
  std::printf("decision=%s reason=%s request_id=%llu attempt=%llu\n", name_of(o.decision), o.reason_code.c_str(),
              (unsigned long long)o.request_id.value(), (unsigned long long)o.attempt_id.value());
}

int main(int argc, char** argv) {
  if (argc < 2) { std::printf("usage: inference_scheduler <cmd> [args]\n  commands: serve worker submit cancel status queue workers stats snapshot explain drain recover bench\n"); return 2; }
  std::string cmd = argv[1];

  if (cmd == "serve") {
    std::string host = argc > 2 ? argv[2] : "127.0.0.1"; int port = argc > 3 ? std::atoi(argv[3]) : 29840;
    std::string c = "inference_scheduler_coordinator " + host + " " + std::to_string(port) + " 2";
    std::printf("serving coordinator: %s\n", c.c_str());
    return std::system(c.c_str());
  }
  if (cmd == "worker") {
    std::string host = argc > 2 ? argv[2] : "127.0.0.1"; int port = argc > 3 ? std::atoi(argv[3]) : 29840;
    std::uint64_t wid = argc > 4 ? std::strtoull(argv[4], nullptr, 10) : 11;
    std::uint64_t bid = argc > 5 ? std::strtoull(argv[5], nullptr, 10) : 1001;
    int units = argc > 6 ? std::atoi(argv[6]) : 8;
    std::string backend = argc > 7 ? argv[7] : "cpu";
    std::string models = argc > 8 ? argv[8] : "";
    std::string c = "inference_scheduler_worker " + host + " " + std::to_string(port) + " " + std::to_string(wid) + " " + std::to_string(bid) + " " + std::to_string(units) + " " + backend;
    if (!models.empty()) c += " " + models;
    return std::system(c.c_str());
  }

  std::string state; int ai = 2;
  SchedulerConfig cfg = cfg_from_args(argc, argv, ai, state);
  Scheduler sched(cfg, std::make_shared<SystemClock>());
  auto reg_worker = [&]() { WorkerRegistration w; w.worker = WorkerId(1); w.node = NodeId(1); w.accelerator = AcceleratorId(1); w.boot_id = WorkerBootId(777); w.state = WorkerState::Online; w.capability.backend = "cpu"; w.capability.device_name = "cpu0"; w.capability.total_capacity_units = 256; w.capability.available_capacity_units = 256; sched.register_worker(w); return w; };
  WorkerRegistration w1 = reg_worker();
  if (!state.empty()) { sched.recover(state); reg_worker(); }

  if (cmd == "submit") { RequestSpec s; s.tenant = TenantId(argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 1); s.model = ModelIdentity(argc > 3 ? std::strtoull(argv[3], nullptr, 10) : 10); s.phase = RequestPhase::Prefill; s.tokens.input = 32; s.tokens.output = 16; s.demand.cost_units = 1.0; auto ad = sched.submit(s); if (ad.ok()) { print_admission(ad.value()); if (!state.empty()) sched.persist(state); return 0; } std::printf("error: [%s] %s\n", error_code_name(ad.error().code), ad.error().message.c_str()); return 1; }
  if (cmd == "cancel") { std::uint64_t rid = argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 0; auto c = sched.cancel(RequestId(rid), CancellationReason::ClientRequested, "cli"); if (c.ok()) { std::printf("cancelled\n"); if (!state.empty()) sched.persist(state); return 0; } std::printf("error: %s\n", c.error().message.c_str()); return 1; }
  if (cmd == "status") { auto st = sched.stats(); std::printf("admitted=%llu completed=%llu cancelled=%llu failed=%llu expired=%llu deferred=%llu\n", (unsigned long long)st.requests_admitted, (unsigned long long)st.requests_completed, (unsigned long long)st.requests_cancelled, (unsigned long long)st.requests_failed, (unsigned long long)st.requests_expired, (unsigned long long)st.requests_deferred); std::printf("current: admitted=%d queued=%d running=%d reserved=%d\n", st.current_admitted, st.current_queued, st.current_running, st.current_reserved); return 0; }
  if (cmd == "queue") { std::printf("queued=%zu\n", sched.queued_count()); return 0; }
  if (cmd == "workers") { auto sn = sched.snapshot(); for (const auto& w : sn.workers) std::printf("worker=%llu backend=%s state=%s avail=%d reserved=%d boot=%llu\n", (unsigned long long)w.worker.value(), w.backend.c_str(), name_of(w.state), w.available_capacity_units, w.reserved_units, (unsigned long long)w.boot_id.value()); return 0; }
  if (cmd == "stats") { auto st = sched.stats(); std::printf("submitted=%llu admitted=%llu completed=%llu cancelled=%llu failed=%llu expired=%llu retries=%llu batches=%llu stale_rejected=%llu\n", (unsigned long long)st.requests_submitted, (unsigned long long)st.requests_admitted, (unsigned long long)st.requests_completed, (unsigned long long)st.requests_cancelled, (unsigned long long)st.requests_failed, (unsigned long long)st.requests_expired, (unsigned long long)st.retries, (unsigned long long)st.batches_formed, (unsigned long long)st.stale_rejected); return 0; }
  if (cmd == "snapshot") { auto sn = sched.snapshot(); std::printf("epoch=%llu workers=%zu tenants=%zu active_requests=%zu\n", (unsigned long long)sn.epoch.value(), sn.workers.size(), sn.tenants.size(), sn.requests.size()); for (const auto& r : sn.requests) std::printf("  req=%llu state=%s tenant=%llu phase=%s\n", (unsigned long long)r.request_id.value(), name_of(r.state), (unsigned long long)r.tenant.value(), name_of(r.phase)); return 0; }
  if (cmd == "explain") { std::uint64_t rid = argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 0; auto ex = sched.explain(RequestId(rid)); if (!ex.ok()) { std::printf("explain error\n"); return 1; } std::printf("request=%llu admission=%s reason=%s\n", (unsigned long long)ex.value().request_id.value(), name_of(ex.value().admission), ex.value().admission_reason_code.c_str()); for (const auto& f : ex.value().ordering_factors) std::printf("  factor: %s\n", f.c_str()); for (const auto& c : ex.value().candidates) if (c.eligible) std::printf("  candidate worker=%llu score=%.3f\n", (unsigned long long)c.worker.value(), c.total); return 0; }
  if (cmd == "drain") { sched.begin_drain("cli"); sched.cancel_all_queued("cli drain"); if (!state.empty()) sched.persist(state); std::printf("draining: queued work cancelled\n"); return 0; }
  if (cmd == "recover") { if (state.empty()) { std::printf("recover requires -state <file>\n"); return 1; } auto rc = sched.recover(state); if (!rc.ok()) { std::printf("recover error: [%s] %s\n", error_code_name(rc.error().code), rc.error().message.c_str()); return 1; } auto st = sched.stats(); std::printf("recovered: admitted=%llu queued=%d\n", (unsigned long long)st.requests_admitted, st.current_queued); return 0; }
  if (cmd == "bench") {
    auto ex = std::make_shared<CpuExecutor>(2000, 400000);
    LocalDriver drv(sched, ex, {w1});
    int n = argc > 2 ? std::atoi(argv[2]) : 10000;
    auto t0 = std::chrono::steady_clock::now();
    for (int k = 0; k < n; ++k) { RequestSpec bs; bs.tenant = TenantId(1 + (k % 3)); bs.model = ModelIdentity(10 + (k % 2)); bs.phase = (k % 2) ? RequestPhase::Prefill : RequestPhase::Decode; bs.tokens.input = 32; bs.tokens.output = 16; bs.demand.cost_units = 1.0; sched.submit(bs, RequestId(1000000 + k)); }
    auto t1 = std::chrono::steady_clock::now();
    drv.run_until_idle(1000000);
    auto t2 = std::chrono::steady_clock::now();
    auto st = sched.stats();
    double submit_s = std::chrono::duration<double>(t1 - t0).count();
    double total_s = std::chrono::duration<double>(t2 - t0).count();
    if (submit_s <= 0.0) submit_s = 1e-9; if (total_s <= 0.0) total_s = 1e-9;
    std::printf("bench: n=%d completed=%llu submit=%.3fs (%.0f/s) total=%.3fs (%.0f completed/s)\n", n, (unsigned long long)st.requests_completed, submit_s, n / submit_s, total_s, st.requests_completed / total_s);
    return 0;
  }
  std::printf("unknown command: %s\n", cmd.c_str());
  return 2;
}