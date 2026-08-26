#include "protocol.hpp"
#include "inference_scheduler/inference_scheduler.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
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

int main(int argc, char** argv) {
  if (argc < 3) { std::printf("usage: client <host> <port>\n"); return 2; }
  std::string host = argv[1]; int port = std::atoi(argv[2]);
  FramedConnection::init();
  auto conn = FramedConnection::connect(host, port);
  if (!conn || !conn->valid()) { std::printf("client: connect failed\n"); return 1; }

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