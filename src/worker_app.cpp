#include "protocol.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>

using namespace inference_scheduler;
using namespace inference_scheduler::net;

int main(int argc, char** argv) {
  if (argc < 7) { std::printf("usage: worker <host> <port> <worker_id> <boot_id> <units> <backend>\n"); return 2; }
  std::string host = argv[1];
  int port = std::atoi(argv[2]);
  std::uint64_t wid = std::strtoull(argv[3], nullptr, 10);
  std::uint64_t bid = std::strtoull(argv[4], nullptr, 10);
  int units = std::atoi(argv[5]);
  std::string backend = argv[6];

  FramedConnection::init();
  auto conn = FramedConnection::connect(host, port);
  if (!conn || !conn->valid()) { std::printf("worker %llu: connect failed\n", (unsigned long long)wid); return 1; }

  Json reg = Json::object();
  reg["op"] = Json::string("register");
  reg["worker"] = id_json(wid);
  reg["boot"] = id_json(bid);
  reg["node"] = id_json(wid * 100 + 1);
  reg["accel"] = id_json(wid * 100 + 2);
  reg["units"] = Json::number(static_cast<double>(units));
  reg["backend"] = Json::string(backend);
  reg["device"] = Json::string(backend + "-" + std::to_string(wid));
  Json marr = Json::array();
  if (argc >= 8) { std::string mm = argv[7]; std::size_t pos = 0; while (pos < mm.size()) { std::size_t cm = mm.find(',', pos); if (cm == std::string::npos) cm = mm.size(); marr.arr.push_back(id_json(std::strtoull(mm.substr(pos, cm - pos).c_str(), nullptr, 10))); pos = (cm == mm.size()) ? mm.size() : cm + 1; } }
  reg["models"] = marr;
  if (!send_json(conn, reg)) { std::printf("worker %llu: register send failed\n", (unsigned long long)wid); return 1; }

  std::set<std::uint64_t> seen;  // request ids already seen (for one-shot fail)
  while (true) {
    Json msg = recv_json(conn);
    if (msg.is_null()) break;
    std::string op = msg.getstr("op");
    if (op == "run") {
      std::vector<WorkRequest> reqs;
      const Json* members = msg.get("members");
      if (members && members->is_arr()) {
        for (const auto& mj : members->arr) {
          WorkRequest w;
          w.request_id = RequestId(num_u64(mj, "req"));
          w.attempt_id = AttemptId(num_u64(mj, "attempt"));
          w.generation = Generation(num_u64(mj, "gen"));
          w.phase = static_cast<RequestPhase>(static_cast<int>(mj.getnum("phase", 1)));
          w.model = ModelIdentity(num_u64(mj, "model"));
          w.revision = ModelRevision(num_u64(mj, "rev"));
          w.adapter = AdapterIdentity(num_u64(mj, "adapter"));
          w.tokens.input = static_cast<std::int64_t>(mj.getnum("tin", 0));
          w.tokens.output = static_cast<std::int64_t>(mj.getnum("tout", 0));
          w.cost_units = static_cast<std::int64_t>(mj.getnum("cost", 1.0));
          w.payload = mj.getstr("payload");
          reqs.push_back(w);
        }
      }
      std::vector<WorkResult> results;
      // One-shot retry: a fail_once request fails the FIRST time it is seen.
      bool have_false_success = false;
      for (const auto& wr : reqs) {
        bool is_fail_once = wr.payload == "fail_once";
        if (is_fail_once && seen.count(wr.request_id.value()) == 0) {
          seen.insert(wr.request_id.value());
          WorkResult fr;
          fr.request_id = wr.request_id; fr.attempt_id = wr.attempt_id; fr.generation = wr.generation;
          fr.status = CompletionStatus::Failed; fr.failure_class = FailureClass::RetryableWorker;
          fr.work_units = wr.cost_units; fr.error_message = "one-shot retryable failure";
          results.push_back(fr);
        } else {
          WorkResult r;
          r.request_id = wr.request_id; r.attempt_id = wr.attempt_id; r.generation = wr.generation;
          r.status = CompletionStatus::Succeeded; r.output_tokens_produced = wr.tokens.output; r.work_units = wr.cost_units;
          r.duration_us = static_cast<std::int64_t>(wr.cost_units * 3);
          results.push_back(r);
        }
      }
      (void)have_false_success;
      Json done = Json::object();
      done["op"] = Json::string("completed");
      done["batch_id"] = id_json(num_u64(msg, "batch_id", 0));
      done["units"] = id_json(static_cast<std::uint64_t>(units));
      Json res_arr = Json::array();
      for (const auto& r : results) {
        Json rj = Json::object();
        rj["req"] = id_json(r.request_id.value());
        rj["attempt"] = id_json(r.attempt_id.value());
        rj["gen"] = id_json(r.generation.value());
        rj["status"] = Json::number(static_cast<int>(r.status));
        rj["failure"] = Json::number(static_cast<int>(r.failure_class));
        rj["out"] = Json::number(static_cast<double>(r.output_tokens_produced));
        rj["work"] = Json::number(static_cast<double>(r.work_units));
        rj["dur"] = Json::number(static_cast<double>(r.duration_us));
        rj["msg"] = Json::string(r.error_message);
        res_arr.arr.push_back(rj);
      }
      done["results"] = res_arr;
      send_json(conn, done);
    } else if (op == "shutdown") { break; }
  }
  conn->close();
  FramedConnection::shutdown();
  std::printf("worker %llu done\n", (unsigned long long)wid);
  return 0;
}