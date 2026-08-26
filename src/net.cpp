#include "net.hpp"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

#include <cstring>

namespace inference_scheduler::net {
namespace {
bool g_init = false;
int g_listen = -1;
bool send_all(int fd, const char* data, std::size_t n) {
  std::size_t off = 0;
  while (off < n) {
#ifdef _WIN32
    int w = ::send(fd, data + off, static_cast<int>(n - off), 0);
#else
    ssize_t w = ::send(fd, data + off, n - off, MSG_NOSIGNAL);
#endif
    if (w <= 0) return false;
    off += static_cast<std::size_t>(w);
  }
  return true;
}
bool recv_all(int fd, char* data, std::size_t n) {
  std::size_t off = 0;
  while (off < n) {
#ifdef _WIN32
    int r = ::recv(fd, data + off, static_cast<int>(n - off), 0);
#else
    ssize_t r = ::recv(fd, data + off, n - off, 0);
#endif
    if (r <= 0) return false;
    off += static_cast<std::size_t>(r);
  }
  return true;
}
void close_fd(int fd) {
#ifdef _WIN32
  if (fd != static_cast<int>(INVALID_SOCKET)) closesocket(static_cast<SOCKET>(fd));
#else
  if (fd >= 0) ::close(fd);
#endif
}
}  // namespace

bool FramedConnection::init() {
#ifdef _WIN32
  if (!g_init) { WSADATA d; if (WSAStartup(MAKEWORD(2, 2), &d) != 0) return false; g_init = true; }
#else
  g_init = true;
#endif
  return true;
}
void FramedConnection::shutdown() {
  if (g_listen != -1) { close_fd(g_listen); g_listen = -1; }
#ifdef _WIN32
  if (g_init) { WSACleanup(); g_init = false; }
#else
  g_init = false;
#endif
}
FramedConnection::FramedConnection(int fd) : fd_(fd) {}
FramedConnection::~FramedConnection() { close(); }
bool FramedConnection::valid() const { return fd_ >= 0; }
void FramedConnection::close() { if (fd_ >= 0) { close_fd(fd_); fd_ = -1; } }

bool FramedConnection::listen(const std::string& host, int port) {
  if (!init()) return false;
  int fd = static_cast<int>(socket(AF_INET, SOCK_STREAM, IPPROTO_TCP));
  if (fd < 0) return false;
  { int reuse = 1; setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char*>(&reuse), sizeof(reuse)); }
  sockaddr_in addr{}; addr.sin_family = AF_INET; addr.sin_port = htons(static_cast<unsigned short>(port));
  if (inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) { close_fd(fd); return false; }
  if (bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) { close_fd(fd); return false; }
  if (::listen(fd, 8) != 0) { close_fd(fd); return false; }
  if (g_listen != -1) close_fd(g_listen);
  g_listen = fd;
  return true;
}

std::shared_ptr<FramedConnection> FramedConnection::accept() {
  if (g_listen == -1) return nullptr;
  sockaddr_in c{};
  std::shared_ptr<FramedConnection> out;
#ifdef _WIN32
  int clen = sizeof(c);
  SOCKET cfd = ::accept(g_listen, reinterpret_cast<sockaddr*>(&c), &clen);
  if (cfd == INVALID_SOCKET) return nullptr;
  out = std::make_shared<FramedConnection>(static_cast<int>(cfd));
#else
  socklen_t clen = sizeof(c);
  int cfd = ::accept(g_listen, reinterpret_cast<sockaddr*>(&c), &clen);
  if (cfd < 0) return nullptr;
  out = std::make_shared<FramedConnection>(cfd);
#endif
  return out;
}

std::shared_ptr<FramedConnection> FramedConnection::connect(const std::string& host, int port) {
  if (!init()) return nullptr;
  int fd = static_cast<int>(socket(AF_INET, SOCK_STREAM, IPPROTO_TCP));
  if (fd < 0) return nullptr;
  sockaddr_in addr{}; addr.sin_family = AF_INET; addr.sin_port = htons(static_cast<unsigned short>(port));
  if (inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) { close_fd(fd); return nullptr; }
  if (::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) { close_fd(fd); return nullptr; }
  return std::make_shared<FramedConnection>(fd);
}

bool FramedConnection::send(std::string_view payload) {
  if (fd_ < 0) return false;
  if (payload.size() > k_max_frame) return false;
  std::uint32_t len = static_cast<std::uint32_t>(payload.size());
  char hdr[4]; std::memcpy(hdr, &len, 4);
  return send_all(fd_, hdr, 4) && send_all(fd_, payload.data(), payload.size());
}

bool FramedConnection::recv(std::string& payload) {
  if (fd_ < 0) return false;
  char hdr[4];
  if (!recv_all(fd_, hdr, 4)) return false;
  std::uint32_t len = 0; std::memcpy(&len, hdr, 4);
  if (len == 0 || len > k_max_frame) return false;
  payload.resize(len);
  return recv_all(fd_, payload.data(), payload.size());
}

}  // namespace inference_scheduler::net