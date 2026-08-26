#include "protocol.hpp"
#include "inference_scheduler/inference_scheduler.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <set>
#include <string>
#include <thread>
#include <vector>

using namespace inference_scheduler;
using namespace inference_scheduler::net;

static int g_fail = 0;
struct Track { std::uint64_t id; std::string model_name; };

Json send_wait(const std::shared_ptr<FramedConnection>& c, const Json& req) {
  if (!send_json(c, req)) return Json::null();
  return recv_json(c);
}


int run_extended(const std::shared_ptr<FramedConnection>& conn, const std::string& scratch) {
  int fail = 0;
  for (int i = 0; i < 200; ++i) { Json r = Json::object(); r["op"] = Json::string("ready"); Json ack = send_wait(conn, r); if (ack.is_obj() && ack.getnum("workers", 0) >= 2.0) break; std::this_thread::sleep_for(std::chrono::milliseconds(50)); }
  auto submit = [&](const RequestSpec& spec, std::uint64_t& out) -> int { Json j = spec_to_json(spec); j["op"] = Json::string("submit"); Json ack = send_wait(conn, j); out = num_u64(ack, "request_id", 0); return static_cast<int>(ack.getnum("decision", 2)); };
  RequestSpec m10; m10.tenant = TenantId(1); m10.model = ModelIdentity(10); m10.revision = ModelRevision(1); m10.phase = RequestPhase::Prefill; m10.latency_class = LatencyClass::Standard; m10.tokens.input = 32; m10.tokens.output = 16; m10.demand.cost_units = 1.0; m10.model_name = "model-a";
  RequestSpec m20; m20.tenant = TenantId(2); m20.model = ModelIdentity(20); m20.revision = ModelRevision(1); m20.phase = RequestPhase::Decode; m20.latency_class = LatencyClass::Interactive; m20.tokens.input = 8; m20.tokens.output = 64; m20.demand.cost_units = 1.0; m20.model_name = "model-b";
  RequestSpec m30; m30.tenant = TenantId(3); m30.model = ModelIdentity(30); m30.revision = ModelRevision(1); m30.phase = RequestPhase::Prefill; m30.latency_class = LatencyClass::Throughput; m30.tokens.input = 64; m30.tokens.output = 8; m30.demand.cost_units = 2.0; m30.model_name = "model-c";
  std::vector<std::uint64_t> pending; std::set<std::uint64_t> done; std::uint64_t rid = 0;
  for (int i = 0; i < 4; ++i) { int d = submit(m10, rid); if (d == 0) pending.push_back(rid); }
  for (int i = 0; i < 4; ++i) { int d = submit(m20, rid); if (d == 0) pending.push_back(rid); }
  for (int i = 0; i < 4; ++i) { int d = submit(m30, rid); if (d == 0) pending.push_back(rid); }
  RequestSpec Ss = m10; Ss.model = ModelIdentity(40); Ss.model_name = "resume"; Ss.payload = "resume";
  std::uint64_t S_id = 0; int ds = submit(Ss, S_id); if (ds != 0) { std::printf("  FAIL: resume submit\n"); conn->close(); FramedConnection::shutdown(); return 1; }
  int round = 0;
  while (!pending.empty() && round < 100) {
    ++round; std::vector<std::uint64_t> next; std::set<std::uint64_t> doneRound;
    for (auto id : pending) { Json r = Json::object(); r["op"] = Json::string("dispatch"); r["request_id"] = id_json(id); Json ack = send_wait(conn, r); int decision = static_cast<int>(ack.getnum("decision", 1)); if (decision == 0) { const Json* members = ack.get("members"); if (members && members->is_arr()) { for (const auto& m : members->arr) { std::uint64_t req = num_u64(m, "req"); std::string reason = m.getstr("reason"); if (reason == "success" || reason == "failed" || reason == "expired" || reason == "cancelled") doneRound.insert(req); else next.push_back(req); } } } else next.push_back(id); }
    pending.clear(); for (auto x : doneRound) done.insert(x); for (auto x : next) pending.push_back(x); if (doneRound.empty() && next.empty()) break;
  }
  { std::ofstream f(scratch + "/resume_id.txt"); f << S_id; }
  std::printf("PHASE_A_DONE resume_id=%llu\n", (unsigned long long)S_id); std::fflush(stdout);
  for (int i = 0; i < 600; ++i) { std::ifstream f(scratch + "/proceed.flag"); if (f.good()) break; std::this_thread::sleep_for(std::chrono::milliseconds(50)); }
  for (int i = 0; i < 200; ++i) { Json r = Json::object(); r["op"] = Json::string("ready"); Json ack = send_wait(conn, r); if (ack.is_obj() && ack.getnum("workers", 0) >= 3.0) break; std::this_thread::sleep_for(std::chrono::milliseconds(50)); }
  Json ri = Json::object(); ri["op"] = Json::string("request_info"); ri["request_id"] = id_json(S_id); Json riAck = send_wait(conn, ri);
  std::uint64_t S_attempt = riAck.is_obj() ? num_u64(riAck, "attempt", 0) : 0;
  std::uint64_t S_gen = riAck.is_obj() ? num_u64(riAck, "generation", 0) : 0;
  Json re = Json::object(); re["op"] = Json::string("roll_epoch"); Json reAck = send_wait(conn, re);
  std::uint64_t curEpoch = reAck.is_obj() ? num_u64(reAck, "epoch", 2) : 2;
  std::uint64_t oldEpoch = curEpoch - 1;
  auto complete = [&](std::uint64_t ep, std::uint64_t wid, std::uint64_t boot, std::uint64_t req, std::uint64_t attempt, std::uint64_t gen) -> Json { Json m = Json::object(); m["op"] = Json::string("complete"); m["epoch"] = id_json(ep); m["worker"] = id_json(wid); m["boot"] = id_json(boot); m["request"] = id_json(req); m["attempt"] = id_json(attempt); m["generation"] = id_json(gen); m["status"] = Json::number(0); m["units"] = Json::number(8); return send_wait(conn, m); };
  Json a1 = complete(oldEpoch, 11, 1001, S_id, S_attempt, S_gen);
  if (a1.is_obj() && static_cast<int>(a1.getnum("acceptance", -1)) == static_cast<int>(CompletionAcceptance::StaleEpoch)) { std::printf("  OK: stale-epoch completion rejected\n"); } else { std::printf("  FAIL: stale-epoch (got %d)\n", static_cast<int>(a1.getnum("acceptance", -1))); ++fail; }
  Json a2 = complete(curEpoch, 11, 1001, S_id, S_attempt, S_gen);
  if (a2.is_obj() && static_cast<int>(a2.getnum("acceptance", -1)) == static_cast<int>(CompletionAcceptance::StaleWorker)) { std::printf("  OK: stale-worker-boot completion rejected\n"); } else { std::printf("  FAIL: stale-worker (got %d)\n", static_cast<int>(a2.getnum("acceptance", -1))); ++fail; }
  Json a3 = complete(curEpoch, 11, 9999, S_id, S_attempt, S_gen + 1);
  if (a3.is_obj() && static_cast<int>(a3.getnum("acceptance", -1)) == static_cast<int>(CompletionAcceptance::StaleAttempt)) { std::printf("  OK: stale-attempt completion rejected\n"); } else { std::printf("  FAIL: stale-attempt (got %d)\n", static_cast<int>(a3.getnum("acceptance", -1))); ++fail; }
  Json dr = Json::object(); dr["op"] = Json::string("dispatch"); dr["request_id"] = id_json(S_id); Json dack = send_wait(conn, dr);
  bool ok_resume = false;
  if (dack.is_obj() && static_cast<int>(dack.getnum("decision", -1)) == 0) { const Json* members = dack.get("members"); if (members && members->is_arr()) { for (const auto& m : members->arr) { if (m.getstr("reason") == "success") ok_resume = true; } } }
  if (ok_resume) { std::printf("  OK: new worker resumed valid work under fresh authority\n"); } else { std::printf("  FAIL: resume dispatch (decision=%d reason=%s worker=%llu)\n", static_cast<int>(dack.getnum("decision", -1)), dack.getstr("reason").c_str(), (unsigned long long)num_u64(dack, "worker", 0)); ++fail; }
  Json sr = Json::object(); sr["op"] = Json::string("stats"); Json st = send_wait(conn, sr);
  double stale = st.getnum("stale_rejected", 0); double comp = st.getnum("completed", 0); double q = st.getnum("current_queued", 0); double rn = st.getnum("current_running", 0);
  std::printf("extended: completed=%.0f stale_rejected=%.0f queued=%.0f running=%.0f\n", comp, stale, q, rn);
  if (stale < 3.0) { std::printf("  FAIL: expected stale_rejected >= 3\n"); ++fail; }
  if (comp < 13.0) { std::printf("  FAIL: expected >= 13 completed (12 + resume)\n"); ++fail; }
  if (q != 0.0 || rn != 0.0) { std::printf("  FAIL: leaked queue/running\n"); ++fail; }
  { Json s = Json::object(); s["op"] = Json::string("shutdown"); send_wait(conn, s); }
  if (fail == 0) { std::printf("CLIENT EXTENDED PASS\n"); conn->close(); FramedConnection::shutdown(); return 0; }
  std::printf("CLIENT EXTENDED FAIL (%d)\n", fail); conn->close(); FramedConnection::shutdown(); return 1;
}
int main(int argc, char** argv) {
  if (argc < 3) { std::printf("usage: client <host> <port>\n"); return 2; }
  std::string host = argv[1]; int port = std::atoi(argv[2]);
  FramedConnection::init();
  auto conn = FramedConnection::connect(host, port);
  if (!conn || !conn->valid()) { std::printf("client: connect failed\n"); return 1; }
  if (argc > 3 && std::string(argv[3]) == "extended") { return run_extended(conn, argc > 4 ? argv[4] : "scratch"); }

  // wait for >= 2 workers
  for (int i = 0; i < 200; ++i) {
    Json r = Json::object(); r["op"] = Json::string("ready");
    Json ack = send_wait(conn, r);
    if (ack.is_obj() && ack.getnum("workers", 0) >= 2.0) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }

  auto submit = [&](const RequestSpec& spec, std::uint64_t& out_id) -> int {
    Json j = spec_to_json(spec); j["op"] = Json::string("submit");
    Json ack = send_wait(conn, j);
    int decision = static_cast<int>(ack.getnum("decision", 2));
    out_id = num_u64(ack, "request_id", 0);
    return decision;
  };

  std::vector<std::uint64_t> pending;
  std::uint64_t rid = 0;

  RequestSpec m10a; m10a.tenant = TenantId(1); m10a.model = ModelIdentity(10); m10a.revision = ModelRevision(1); m10a.phase = RequestPhase::Prefill; m10a.latency_class = LatencyClass::Standard; m10a.tokens.input = 32; m10a.tokens.output = 16; m10a.demand.cost_units = 1.0; m10a.model_name = "model-a";
  RequestSpec m20a; m20a.tenant = TenantId(2); m20a.model = ModelIdentity(20); m20a.revision = ModelRevision(1); m20a.phase = RequestPhase::Decode; m20a.latency_class = LatencyClass::Interactive; m20a.tokens.input = 8; m20a.tokens.output = 64; m20a.demand.cost_units = 1.0; m20a.model_name = "model-b";
  RequestSpec m30a; m30a.tenant = TenantId(3); m30a.model = ModelIdentity(30); m30a.revision = ModelRevision(1); m30a.phase = RequestPhase::Prefill; m30a.latency_class = LatencyClass::Throughput; m30a.tokens.input = 64; m30a.tokens.output = 8; m30a.demand.cost_units = 2.0; m30a.model_name = "model-c";

  // model-a x 4 (worker 1), model-b x 4 (worker 2), model-c x 4 (worker 1)
  std::set<std::uint64_t> admitted;
  for (int i = 0; i < 4; ++i) { int d = submit(m10a, rid); if (d == 0) { admitted.insert(rid); pending.push_back(rid); } }
  for (int i = 0; i < 4; ++i) { int d = submit(m20a, rid); if (d == 0) { admitted.insert(rid); pending.push_back(rid); } }
  for (int i = 0; i < 4; ++i) { int d = submit(m30a, rid); if (d == 0) { admitted.insert(rid); pending.push_back(rid); } }

  // one-shot retry request
  RequestSpec fr = m10a; fr.payload = "fail_once"; fr.model_name = "failonce";
  std::uint64_t fr_id = 0; int d = submit(fr, fr_id); if (d == 0) { admitted.insert(fr_id); pending.push_back(fr_id); }

  // a request to cancel (queued)
  std::uint64_t cancel_id = 0; d = submit(m20a, cancel_id); if (d == 0) { admitted.insert(cancel_id); }
  { Json r = Json::object(); r["op"] = Json::string("cancel"); r["request_id"] = id_json(cancel_id); Json ack = send_wait(conn, r); if (!ack.getbool("ok", false)) { std::printf("  FAIL: cancel\n"); ++g_fail; } }

  const int total_admitted = static_cast<int>(admitted.size());
  // dispatch loop
  int round = 0;
  while (!pending.empty() && round < 100) {
    ++round;
    std::vector<std::uint64_t> next;
    std::set<std::uint64_t> done_this_round;
    for (auto id : pending) {
      Json r = Json::object(); r["op"] = Json::string("dispatch"); r["request_id"] = id_json(id); r["explain"] = Json::boolean(true);
      Json ack = send_wait(conn, r);
      int decision = static_cast<int>(ack.getnum("decision", 1));
      if (decision == 0) {  // dispatched
        const Json* members = ack.get("members");
        if (members && members->is_arr()) {
          for (const auto& m : members->arr) {
            std::uint64_t req = num_u64(m, "req");
            std::string reason = m.getstr("reason");
            if (reason == "success" || reason == "failed" || reason == "expired" || reason == "cancelled") { done_this_round.insert(req); }
            else { next.push_back(req); }
          }
        }
      } else { next.push_back(id); }  // Hold / NoEligibleWorker -> retry later
    }
    pending.clear();
    for (auto x : done_this_round) if (admitted.count(x)) admitted.erase(x);
    for (auto x : next) if (admitted.count(x)) pending.push_back(x);
    if (done_this_round.empty() && next.empty()) break;
  }

  // query stats
  Json r = Json::object(); r["op"] = Json::string("stats");
  Json st = send_wait(conn, r);
  double completed = st.getnum("completed", -1);
  double cancelled = st.getnum("cancelled", 0);
  int total = total_admitted;
  std::printf("client: total_admitted=%d completed=%.0f cancelled=%.0f queued=%.0f running=%.0f\n", total, completed, cancelled, st.getnum("current_queued",0), st.getnum("current_running",0));
  if (st.getnum("current_queued", -1) != 0 || st.getnum("current_running", -1) != 0) { std::printf("  FAIL: leaked queue/running\n"); ++g_fail; }
  if (cancelled >= 1.0) { std::printf("  OK: cancellation observed\n"); } else { std::printf("  FAIL: cancellation not observed\n"); ++g_fail; }
  if (completed + cancelled >= (double)total) { std::printf("  OK: all admitted resolved\n"); } else { std::printf("  FAIL: not all admitted resolved (completed=%.0f cancelled=%.0f total=%d)\n", completed, cancelled, total); ++g_fail; }

  { Json s = Json::object(); s["op"] = Json::string("shutdown"); send_wait(conn, s); }
  if (g_fail == 0) { std::printf("CLIENT PASS\n"); conn->close(); FramedConnection::shutdown(); return 0; }
  std::printf("CLIENT FAIL (%d)\n", g_fail);
  conn->close(); FramedConnection::shutdown();
  return 1;
}