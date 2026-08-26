#include "persistence.hpp"

#include <cstdio>
#include <cstring>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace inference_scheduler::detail {

namespace {

std::uint64_t fnv1a(std::uint64_t h, std::uint64_t b) noexcept {
  h ^= b;
  h *= 1099511628211ull;
  return h;
}

std::uint64_t checksum_of(std::string_view s) {
  std::uint64_t h = 1469598103934665603ull;
  for (char c : s) h = fnv1a(h, static_cast<std::uint8_t>(c));
  return h;
}

void put_u32(std::string& s, std::uint32_t v) {
  s.append(reinterpret_cast<const char*>(&v), 4);
}
void put_u64(std::string& s, std::uint64_t v) {
  s.append(reinterpret_cast<const char*>(&v), 8);
}

bool get_u32(std::string_view s, std::size_t& off, std::uint32_t& out) {
  if (off + 4 > s.size()) return false;
  std::memcpy(&out, s.data() + off, 4);
  off += 4;
  return true;
}
bool get_u64(std::string_view s, std::size_t& off, std::uint64_t& out) {
  if (off + 8 > s.size()) return false;
  std::memcpy(&out, s.data() + off, 8);
  off += 8;
  return true;
}

}  // namespace

Result<void> write_persistence_file(const std::string& path, std::string_view payload) {
  std::string frame;
  frame.reserve(k_persistence_header_size + payload.size());
  put_u32(frame, k_persistence_magic);
  put_u32(frame, k_persistence_frame_version);
  put_u64(frame, static_cast<std::uint64_t>(payload.size()));
  frame.append(payload.data(), payload.size());
  put_u64(frame, checksum_of(frame));

  const std::string tmp = path + ".tmp";
#ifdef _WIN32
  HANDLE h = CreateFileA(tmp.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS,
                         FILE_ATTRIBUTE_NORMAL, nullptr);
  if (h == INVALID_HANDLE_VALUE) {
    return make_error(ErrorCode::PersistenceFailure, "create temp failed",
                      "win32=" + std::to_string(GetLastError()));
  }
  DWORD written = 0;
  const BOOL ok =
      WriteFile(h, frame.data(), static_cast<DWORD>(frame.size()), &written, nullptr);
  const std::uint64_t target = static_cast<std::uint64_t>(frame.size());
  const BOOL flushed = FlushFileBuffers(h);
  const DWORD werr = GetLastError();
  CloseHandle(h);
  if (!ok || static_cast<std::uint64_t>(written) != target) {
    return make_error(ErrorCode::PersistenceFailure, "write failed",
                      "win32=" + std::to_string(werr));
  }
  if (!flushed) {
    return make_error(ErrorCode::PersistenceFailure, "flush failed",
                      "win32=" + std::to_string(werr));
  }
  if (!MoveFileExA(tmp.c_str(), path.c_str(),
                   MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
    return make_error(ErrorCode::PersistenceFailure, "atomic replace failed",
                      "win32=" + std::to_string(GetLastError()));
  }
#else
  {
    std::FILE* f = std::fopen(tmp.c_str(), "wb");
    if (!f) return make_error(ErrorCode::PersistenceFailure, "open temp failed");
    const std::size_t w = std::fwrite(frame.data(), 1, frame.size(), f);
    const int ec = std::fflush(f);
    std::fclose(f);
    if (w != frame.size() || ec != 0) {
      return make_error(ErrorCode::PersistenceFailure, "write/flush failed");
    }
  }
  if (std::rename(tmp.c_str(), path.c_str()) != 0) {
    return make_error(ErrorCode::PersistenceFailure, "atomic replace failed");
  }
#endif
  return Result<void>::success();
}

Result<std::string> read_persistence_file(const std::string& path) {
  std::vector<char> blob;
#ifdef _WIN32
  HANDLE h = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                         OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (h == INVALID_HANDLE_VALUE) {
    return make_error(ErrorCode::NotFound, "persistence file not readable",
                      "win32=" + std::to_string(GetLastError()));
  }
  LARGE_INTEGER sz{};
  if (!GetFileSizeEx(h, &sz) || sz.QuadPart <= 0) {
    CloseHandle(h);
    return make_error(ErrorCode::Corruption, "unreadable size");
  }
  blob.resize(static_cast<std::size_t>(sz.QuadPart));
  DWORD rd = 0;
  const BOOL ok = ReadFile(h, blob.data(), static_cast<DWORD>(blob.size()), &rd, nullptr);
  const DWORD werr = GetLastError();
  CloseHandle(h);
  if (!ok) return make_error(ErrorCode::PersistenceFailure, "read failed",
                             "win32=" + std::to_string(werr));
  blob.resize(rd);
#else
  std::FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) return make_error(ErrorCode::NotFound, "persistence file not readable");
  std::fseek(f, 0, SEEK_END);
  const long sz = std::ftell(f);
  if (sz <= 0) { std::fclose(f); return make_error(ErrorCode::Corruption, "unreadable size"); }
  std::fseek(f, 0, SEEK_SET);
  blob.resize(static_cast<std::size_t>(sz));
  const std::size_t rd = std::fread(blob.data(), 1, blob.size(), f);
  std::fclose(f);
  if (rd != blob.size()) return make_error(ErrorCode::PersistenceFailure, "read failed");
#endif

  const std::string_view s(blob.data(), blob.size());
  if (s.size() < k_persistence_header_size + k_persistence_trailer_size) {
    return make_error(ErrorCode::Corruption, "envelope too short",
                      "size=" + std::to_string(s.size()));
  }
  std::size_t off = 0;
  std::uint32_t magic = 0;
  std::uint32_t ver = 0;
  std::uint64_t len = 0;
  std::uint64_t chk = 0;
  if (!get_u32(s, off, magic) || !get_u32(s, off, ver) || !get_u64(s, off, len)) {
    return make_error(ErrorCode::Corruption, "envelope fields truncated");
  }
  // The checksum is a trailing 8-byte trailer, not adjacent to the header.
  std::memcpy(&chk, s.data() + s.size() - k_persistence_trailer_size, k_persistence_trailer_size);
  if (magic != k_persistence_magic) return make_error(ErrorCode::Corruption, "bad magic");
  if (ver != k_persistence_frame_version) {
    return make_error(ErrorCode::Corruption, "unsupported frame version",
                      "version=" + std::to_string(ver));
  }
  const std::size_t payload_len = static_cast<std::size_t>(len);
  if (payload_len + k_persistence_header_size + k_persistence_trailer_size != s.size()) {
    return make_error(ErrorCode::Corruption, "length mismatch",
                      "len=" + std::to_string(len) + " vs " + std::to_string(s.size()));
  }
  const std::string_view payload = s.substr(k_persistence_header_size, payload_len);
  const std::string_view prefix = s.substr(0, k_persistence_header_size + payload_len);
  if (checksum_of(prefix) != chk) {
    return make_error(ErrorCode::Corruption, "checksum mismatch");
  }
  return Result<std::string>::ok(std::string(payload));
}

}  // namespace inference_scheduler::detail