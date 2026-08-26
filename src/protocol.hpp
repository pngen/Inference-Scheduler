#pragma once

#include "json_util.hpp"
#include "net.hpp"
#include "inference_scheduler/executor.hpp"

#include <cstdlib>
#include <memory>

namespace inference_scheduler::net {

inline bool send_json(const std::shared_ptr<FramedConnection>& c, const Json& j) {
  std::string s = dump_json(j);
  return c && c->send(s);
}
inline Json recv_json(const std::shared_ptr<FramedConnection>& c) {
  std::string s;
  if (!c || !c->recv(s)) return Json::null();
  bool ok = true;
  Json j = parse_json(s, ok);
  return ok ? j : Json::null();
}

inline Json id_json(std::uint64_t v) { return Json::string(std::to_string(v)); }
inline std::uint64_t jsu64(const Json& v, std::uint64_t dflt = 0) {
  if (v.is_str()) return std::strtoull(v.str.c_str(), nullptr, 10);
  if (v.is_num()) return static_cast<std::uint64_t>(v.num);
  return dflt;
}
inline std::uint64_t num_u64(const Json& j, const char* k, std::uint64_t dflt = 0) {
  const Json* f = j.get(k);
  if (!f) return dflt;
  if (f->is_str()) return std::strtoull(f->str.c_str(), nullptr, 10);
  if (f->is_num()) return static_cast<std::uint64_t>(f->num);
  return dflt;
}

inline Json spec_to_json(const RequestSpec& s) {
  Json j = Json::object();
  j["kind"] = Json::string("spec");
  j["tenant"] = id_json(s.tenant.value());
  j["session"] = id_json(s.session.value());
  j["sequence"] = id_json(s.sequence.value());
  j["model"] = id_json(s.model.value());
  j["revision"] = id_json(s.revision.value());
  j["adapter"] = id_json(s.adapter.value());
  j["phase"] = Json::number(static_cast<int>(s.phase));
  j["priority"] = Json::number(static_cast<int>(s.priority));
  j["latency"] = Json::number(static_cast<int>(s.latency_class));
  j["tin"] = Json::number(static_cast<double>(s.tokens.input));
  j["tout"] = Json::number(static_cast<double>(s.tokens.output));
  j["tdec"] = Json::number(static_cast<double>(s.tokens.decoded));
  j["cost"] = Json::number(s.demand.cost_units);
  j["mem"] = Json::number(static_cast<double>(s.estimated_memory_bytes));
  j["deadline_ms"] = Json::number(static_cast<double>(s.deadline.valid() ? s.deadline.remaining(ClockTime{}).count() : -1));
  j["maxbatch"] = Json::number(static_cast<double>(s.max_batch_size));
  j["payload"] = Json::string(s.payload);
  return j;
}
inline RequestSpec spec_from_json(const Json& j) {
  RequestSpec s;
  s.tenant = TenantId(num_u64(j, "tenant"));
  s.session = SessionId(num_u64(j, "session"));
  s.sequence = SequenceId(num_u64(j, "sequence"));
  s.model = ModelIdentity(num_u64(j, "model"));
  s.revision = ModelRevision(num_u64(j, "revision"));
  s.adapter = AdapterIdentity(num_u64(j, "adapter"));
  s.phase = static_cast<RequestPhase>(static_cast<int>(j.getnum("phase", 1)));
  s.priority = static_cast<PriorityClass>(static_cast<int>(j.getnum("priority", 2)));
  s.latency_class = static_cast<LatencyClass>(static_cast<int>(j.getnum("latency", 2)));
  s.tokens.input = static_cast<std::int64_t>(j.getnum("tin", 0));
  s.tokens.output = static_cast<std::int64_t>(j.getnum("tout", 0));
  s.tokens.decoded = static_cast<std::int64_t>(j.getnum("tdec", 0));
  s.demand.cost_units = j.getnum("cost", 1.0);
  s.estimated_memory_bytes = static_cast<std::int64_t>(j.getnum("mem", 0));
  s.max_batch_size = static_cast<int>(j.getnum("maxbatch", 0));
  s.payload = j.getstr("payload");
  return s;
}

}  // namespace inference_scheduler::net