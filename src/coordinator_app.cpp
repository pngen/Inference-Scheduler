#include "protocol.hpp"
#include "inference_scheduler/inference_scheduler.hpp"

#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

using namespace inference_scheduler;
using namespace inference_scheduler::net;

struct WorkerSlot {
  std::mutex mu;
  std::condition_variable cv;
  std::shared_ptr<FramedConnection> conn;
  WorkerId worker;
  WorkerBootId boot;
  int units = 1;
  bool has_result = false;
  Json result;
};

struct Coordinator {
  Scheduler sched;
  std::mutex wm;
  std::vector<std::shared_ptr<WorkerSlot>> workers;
  std::map<WorkerId, std::shared_ptr<WorkerSlot>> by_worker;
  explicit Coordinator(SchedulerConfig cfg) : sched(std::move(cfg)) {}
  int worker_count() { std::lock_guard<std::mutex> g(wm); return static_cast<int>(workers.size()); }
  std::shared_ptr<WorkerSlot> find(WorkerId w) { std::lock_guard<std::mutex> g(wm); auto it = by_worker.find(w); return it == by_worker.end() ? nullptr : it->second; }
};

static Coordinator* g_ctx = nullptr;

void worker_thread(std::shared_ptr<WorkerSlot> slot) {
  while (true) {
    Json msg = recv_json(slot->conn);
    if (msg.is_null()) break;
    std::string op = msg.getstr("op");
    if (op == "completed") {
      std::uint64_t batch = num_u64(msg, "batch_id");
      int avail = static_cast<int>(num_u64(msg, "units", 0));
      Json ack = Json::object();
      ack["op"] = Json::string("dispatch_done");
      ack["batch_id"] = id_json(batch);
      Json resarr = Json::array();
      const Json* results = msg.get("results");
      if (results && results->is_arr()) {
        for (const auto& rj : results->arr) {
          CompletionReport start;
          start.epoch = g_ctx->sched.epoch();
          start.worker = slot->worker;
          start.boot_id = slot->boot;
          start.request_id = RequestId(num_u64(rj, "req"));
          start.attempt_id = AttemptId(num_u64(rj, "attempt"));
          start.generation = Generation(num_u64(rj, "gen"));
          start.worker_snapshot.available_capacity_units = avail;
          start.worker_snapshot.total_capacity_units = avail;
          g_ctx->sched.confirm_dispatch_start(start);
        }
        for (const auto& rj : results->arr) {
          CompletionReport c;
          c.epoch = g_ctx->sched.epoch();
          c.worker = slot->worker;
          c.boot_id = slot->boot;
          c.request_id = RequestId(num_u64(rj, "req"));
          c.attempt_id = AttemptId(num_u64(rj, "attempt"));
          c.generation = Generation(num_u64(rj, "gen"));
          c.status = static_cast<CompletionStatus>(static_cast<int>(rj.getnum("status", 0)));
          c.failure_class = static_cast<FailureClass>(static_cast<int>(rj.getnum("failure", 8)));
          c.output_tokens_produced = static_cast<std::int64_t>(rj.getnum("out", 0));
          c.work_units = static_cast<std::int64_t>(rj.getnum("work", 0));
          c.duration_us = static_cast<std::int64_t>(rj.getnum("dur", 0));
          c.error_message = rj.getstr("msg");
          c.worker_snapshot.available_capacity_units = avail;
          c.worker_snapshot.total_capacity_units = avail;
          auto co = g_ctx->sched.complete_attempt(c);
          Json rj2 = Json::object();
          rj2["req"] = id_json(c.request_id.value());
          rj2["acceptance"] = Json::number(static_cast<int>(co.ok() ? co.value().acceptance : CompletionAcceptance::Invalid));
          rj2["reason"] = Json::string(co.ok() ? co.value().reason_code : "error");
          resarr.arr.push_back(rj2);
        }
      }
      ack["results"] = resarr;
      { std::lock_guard<std::mutex> g(slot->mu); slot->result = ack; slot->has_result = true; }
      slot->cv.notify_all();
    } else if (op == "shutdown") { break; }
  }
}

void handle_one(const Json& req, const std::shared_ptr<FramedConnection>& conn) {
  std::string op = req.getstr("op");
  if (op == "ready") { Json ack = Json::object(); ack["op"] = Json::string("ready_ack"); ack["workers"] = Json::number(static_cast<double>(g_ctx->worker_count())); send_json(conn, ack); }
  else if (op == "submit") {
    RequestSpec spec = spec_from_json(req); auto ad = g_ctx->sched.submit(spec);
    Json ack = Json::object(); ack["op"] = Json::string("submit_ack");
    if (ad.ok()) { ack["decision"] = Json::number(static_cast<int>(ad.value().decision)); ack["reason"] = Json::string(ad.value().reason_code); ack["request_id"] = id_json(ad.value().request_id.value()); ack["attempt"] = id_json(ad.value().attempt_id.value()); ack["generation"] = id_json(ad.value().generation.value()); }
    else { ack["decision"] = Json::number(static_cast<int>(AdmissionDecision::Reject)); ack["reason"] = Json::string(ad.error().message); ack["request_id"] = Json::number(0); }
    send_json(conn, ack);
  }
  else if (op == "dispatch") {
    RequestId rid(num_u64(req, "request_id")); bool explain = req.getbool("explain", false);
    auto out = g_ctx->sched.plan_dispatch(rid, explain);
    Json ack = Json::object(); ack["op"] = Json::string("dispatch_ack"); ack["request_id"] = id_json(rid.value());
    if (!out.ok() || out.value().decision != DispatchDecision::Dispatch) { ack["decision"] = Json::number(static_cast<int>(out.ok() ? out.value().decision : DispatchDecision::Hold)); ack["reason"] = Json::string(out.ok() ? out.value().reason_code : out.error().message); ack["members"] = Json::array(); send_json(conn, ack); return; }
    const DispatchOutcome& o = out.value();
    auto slot = g_ctx->find(o.worker_id);
    if (!slot) { ack["decision"] = Json::number(static_cast<int>(DispatchDecision::NoEligibleWorker)); ack["reason"] = Json::string("worker_gone"); ack["members"] = Json::array(); send_json(conn, ack); return; }
    { std::lock_guard<std::mutex> g(slot->mu); slot->has_result = false; }
    Json runm = Json::object(); runm["op"] = Json::string("run"); runm["batch_id"] = id_json(o.batch_id.value()); Json members = Json::array();
    for (auto mid : o.batch_members) {
      auto snap = g_ctx->sched.request_snapshot(mid); auto spr = g_ctx->sched.request_spec(mid);
      Json mj = Json::object(); mj["req"] = id_json(mid.value()); mj["attempt"] = id_json(snap.ok() ? snap.value().attempt_id.value() : 0); mj["gen"] = id_json(snap.ok() ? snap.value().generation.value() : 0);
      mj["phase"] = Json::number(static_cast<int>(snap.ok() ? snap.value().phase : RequestPhase::Prefill)); mj["model"] = id_json(snap.ok() ? snap.value().model.value() : 0); mj["rev"] = id_json(snap.ok() ? snap.value().revision.value() : 0);
      mj["adapter"] = id_json(0); mj["tin"] = Json::number(static_cast<double>(spr.ok() ? spr.value().tokens.input : 0)); mj["tout"] = Json::number(static_cast<double>(spr.ok() ? spr.value().tokens.output : 0));
      mj["cost"] = Json::number(spr.ok() ? spr.value().demand.cost_units : 1.0); mj["payload"] = Json::string(spr.ok() ? spr.value().payload : "");
      members.arr.push_back(mj);
    }
    runm["members"] = members;
    send_json(slot->conn, runm);
    Json done; { std::unique_lock<std::mutex> lk(slot->mu); slot->cv.wait(lk, [&]{ return slot->has_result; }); done = slot->result; }
    ack["decision"] = Json::number(static_cast<int>(DispatchDecision::Dispatch)); ack["worker"] = id_json(o.worker_id.value()); ack["batch"] = id_json(o.batch_id.value()); ack["reason"] = Json::string("dispatched"); ack["members"] = done.get("results") ? done["results"] : Json::array(); ack["done"] = done;
    send_json(conn, ack);
  }
  else if (op == "cancel") { RequestId rid(num_u64(req, "request_id")); auto cr = g_ctx->sched.cancel(rid, CancellationReason::ClientRequested, "client"); Json ack = Json::object(); ack["op"] = Json::string("cancel_ack"); ack["ok"] = Json::boolean(cr.ok()); ack["reason"] = Json::string(cr.ok() ? "cancelled" : cr.error().message); send_json(conn, ack); }
  else if (op == "stats") { auto st = g_ctx->sched.stats(); Json ack = Json::object(); ack["op"] = Json::string("stats_ack"); ack["admitted"] = Json::number(static_cast<double>(st.requests_admitted)); ack["completed"] = Json::number(static_cast<double>(st.requests_completed)); ack["cancelled"] = Json::number(static_cast<double>(st.requests_cancelled)); ack["expired"] = Json::number(static_cast<double>(st.requests_expired)); ack["failed"] = Json::number(static_cast<double>(st.requests_failed)); ack["current_queued"] = Json::number(static_cast<double>(st.current_queued)); ack["current_running"] = Json::number(static_cast<double>(st.current_running)); ack["current_admitted"] = Json::number(static_cast<double>(st.current_admitted)); send_json(conn, ack); }
  else if (op == "shutdown") { Json ack = Json::object(); ack["op"] = Json::string("bye"); send_json(conn, ack); }
}

void client_loop(std::shared_ptr<FramedConnection> conn, Json first) {
  Json req = first;
  while (!req.is_null()) {
    handle_one(req, conn);
    if (req.getstr("op") == "shutdown") break;
    req = recv_json(conn);
  }
}

int main(int argc, char** argv) {
  if (argc < 4) { std::printf("usage: coordinator <host> <port> <min_workers>\n"); return 2; }
  std::string host = argv[1]; int port = std::atoi(argv[2]); int min_workers = std::atoi(argv[3]);
  (void)min_workers;
  FramedConnection::init();
  SchedulerConfig cfg = default_scheduler_config();
  Coordinator ctx(std::move(cfg));
  g_ctx = &ctx;
  if (!FramedConnection::listen(host, port)) { std::printf("coordinator: listen failed\n"); return 1; }
  std::printf("coordinator listening on %s:%d\n", host.c_str(), port);
  while (true) {
    auto conn = FramedConnection::accept();
    if (!conn) continue;
    Json first = recv_json(conn);
    if (first.is_null()) continue;
    if (first.getstr("op") == "register") {
      auto slot = std::make_shared<WorkerSlot>();
      slot->conn = conn; slot->worker = WorkerId(num_u64(first, "worker")); slot->boot = WorkerBootId(num_u64(first, "boot"));
      slot->units = static_cast<int>(num_u64(first, "units", 1));
      WorkerRegistration reg;
      reg.worker = slot->worker; reg.node = NodeId(num_u64(first, "node")); reg.accelerator = AcceleratorId(num_u64(first, "accel")); reg.boot_id = slot->boot;
      reg.state = WorkerState::Online; reg.capability.backend = first.getstr("backend", "cpu"); reg.capability.device_name = first.getstr("device", "cpu");
      reg.capability.total_capacity_units = slot->units; reg.capability.available_capacity_units = slot->units;
      if (const Json* models = first.get("models")) { for (const auto& mm : models->arr) { reg.capability.models.push_back(ModelIdentity(jsu64(mm))); } }
      g_ctx->sched.register_worker(reg);
      { std::lock_guard<std::mutex> g(g_ctx->wm); g_ctx->workers.push_back(slot); g_ctx->by_worker[slot->worker] = slot; }
      std::printf("coordinator: registered worker %llu\n", (unsigned long long)slot->worker.value());
      std::thread(worker_thread, slot).detach();
    } else {
      std::thread([conn, first]() { client_loop(conn, first); }).detach();
    }
  }
  FramedConnection::shutdown();
  return 0;
}