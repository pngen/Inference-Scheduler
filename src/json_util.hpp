#pragma once

#include <cctype>
#include <cstdlib>
#include <map>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace inference_scheduler::net {

// Minimal, bounded JSON value + parser + serializer. Internal to the
// distributed proof harness. Handles the message vocabulary used here.
struct Json {
  enum class Kind { Null, Bool, Num, Str, Arr, Obj };
  Kind kind = Kind::Null;
  bool b = false;
  double num = 0.0;
  std::string str;
  std::vector<Json> arr;
  std::map<std::string, Json> obj;

  static Json null() { return Json{}; }
  static Json boolean(bool v) { Json j; j.kind = Kind::Bool; j.b = v; return j; }
  static Json number(double v) { Json j; j.kind = Kind::Num; j.num = v; return j; }
  static Json string(std::string v) { Json j; j.kind = Kind::Str; j.str = std::move(v); return j; }
  static Json array() { Json j; j.kind = Kind::Arr; return j; }
  static Json object() { Json j; j.kind = Kind::Obj; return j; }

  bool is_null() const { return kind == Kind::Null; }
  bool is_bool() const { return kind == Kind::Bool; }
  bool is_num() const { return kind == Kind::Num; }
  bool is_str() const { return kind == Kind::Str; }
  bool is_arr() const { return kind == Kind::Arr; }
  bool is_obj() const { return kind == Kind::Obj; }
  Json& operator[](const std::string& k) { auto it = obj.find(k); if (it == obj.end()) { it = obj.emplace(k, Json{}).first; } return it->second; }
  const Json* get(const std::string& k) const { auto it = obj.find(k); return it == obj.end() ? nullptr : &it->second; }
  std::string getstr(const std::string& k, const std::string& dflt = "") const { const Json* j = get(k); return (j && j->is_str()) ? j->str : dflt; }
  double getnum(const std::string& k, double dflt = 0) const { const Json* j = get(k); return (j && j->is_num()) ? j->num : dflt; }
  bool getbool(const std::string& k, bool dflt = false) const { const Json* j = get(k); return (j && j->is_bool()) ? j->b : dflt; }
};

namespace detail {
inline bool ws(char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; }
inline void toks(std::string_view s, std::size_t& i) { while (i < s.size() && ws(s[i])) ++i; }
}  // namespace detail


inline std::string dump_json(const Json& j) {
  switch (j.kind) {
    case Json::Kind::Null: return "null";
    case Json::Kind::Bool: return j.b ? "true" : "false";
    case Json::Kind::Num: { std::ostringstream o; o.precision(20); o << j.num; return o.str(); }
    case Json::Kind::Str: { std::string o = "\""; for (char c : j.str) { if (c == '"' || c == '\\') o.push_back('\\'); o.push_back(c); } o.push_back('"'); return o; }
    case Json::Kind::Arr: { std::string o = "["; bool first = true; for (const auto& e : j.arr) { if (!first) o += ","; first = false; o += dump_json(e); } o += "]"; return o; }
    case Json::Kind::Obj: { std::string o = "{"; bool first = true; for (const auto& kv : j.obj) { if (!first) o += ","; first = false; o += "\"" + kv.first + "\":" + dump_json(kv.second); } o += "}"; return o; }
  }
  return "null";
}

// Parses a single JSON value. On malformed input ok=false and null is returned.
inline Json parse_json(std::string_view s, bool& ok) {
  struct Parser {
    std::string_view src; bool& ok;
    std::size_t i = 0;
    void skip() { while (i < src.size() && detail::ws(src[i])) ++i; }
    bool fail() { ok = false; return false; }
    Json value() { skip(); if (i >= src.size()) { ok = false; return Json::null(); } char c = src[i]; if (c == '{') return object(); if (c == '[') return array(); if (c == '"') return Json::string(Jstr()); if (c == 't' || c == 'f') return boolean(); if (c == 'n') { expect("null"); return Json::null(); } return number(); }
    bool expect(std::string_view t) { for (char c : t) { if (i >= src.size() || src[i] != c) return fail(); ++i; } return true; }
    Json object() { ++i; Json j = Json::object(); skip(); if (i < src.size() && src[i] == '}') { ++i; return j; } while (i < src.size()) { skip(); if (i >= src.size() || src[i] != '"') { fail(); return Json::object(); } std::string k = Jstr(); skip(); if (i >= src.size() || src[i] != ':') { fail(); return Json::object(); } ++i; j.obj[std::move(k)] = value(); skip(); if (i < src.size() && src[i] == ',') { ++i; continue; } if (i < src.size() && src[i] == '}') { ++i; return j; } fail(); return Json::object(); } ok = false; return Json::object(); }
    Json array() { ++i; Json j = Json::array(); skip(); if (i < src.size() && src[i] == ']') { ++i; return j; } while (i < src.size()) { j.arr.push_back(value()); skip(); if (i < src.size() && src[i] == ',') { ++i; continue; } if (i < src.size() && src[i] == ']') { ++i; return j; } fail(); return j; } ok = false; return j; }
    std::string Jstr() { ++i; std::string out; while (i < src.size() && src[i] != '"') { char c = src[i]; if (c == '\\') { ++i; if (i >= src.size()) { fail(); return out; } char e = src[i]; switch (e) { case 'n': out.push_back('\n'); break; case 't': out.push_back('\t'); break; case 'r': out.push_back('\r'); break; case 'b': out.push_back('\b'); break; case 'f': out.push_back('\f'); break; default: out.push_back(e); } } else { out.push_back(c); } ++i; } if (i >= src.size()) { fail(); return out; } ++i; return out; }
    Json boolean() { if (src.compare(i, 4, "true") == 0) { i += 4; return Json::boolean(true); } if (src.compare(i, 5, "false") == 0) { i += 5; return Json::boolean(false); } fail(); return Json::boolean(false); }
    Json number() { std::size_t st = i; if (i < src.size() && src[i] == '-') ++i; while (i < src.size() && (std::isdigit(static_cast<unsigned char>(src[i])) || src[i] == '.' || src[i] == 'e' || src[i] == 'E' || src[i] == '+')) ++i; if (i == st) { fail(); return Json::number(0); } return Json::number(std::strtod(std::string(src.substr(st, i - st)).c_str(), nullptr)); }
  };
  Parser p{s, ok};
  Json v = p.value();
  p.skip();
  if (p.i != s.size()) ok = false;
  return v;
}

}  // namespace inference_scheduler::net