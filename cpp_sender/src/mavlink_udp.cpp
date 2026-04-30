#include "mavlink_udp.h"
#include <cstring>
#include <stdexcept>

namespace mavlink_bridge {

MavlinkUDP::MavlinkUDP()
#ifdef WIN32
    : socket_(INVALID_SOCKET)
    , winsockInitialized_(false)
#else
    : socket_(-1)
    , winsockInitialized_(false)
#endif
    , connected_(false)
{
    std::memset(&targetAddr_, 0, sizeof(targetAddr_));
}

MavlinkUDP::~MavlinkUDP() {
    disconnect();
    cleanupWinsock();
}

bool MavlinkUDP::initWinsock() {
#ifdef WIN32
    if (winsockInitialized_) return true;

    int result = WSAStartup(MAKEWORD(2, 2), &wsaData_);
    if (result != 0) {
        return false;
    }
    winsockInitialized_ = true;
    return true;
#else
    winsockInitialized_ = true;
    return true;
#endif
}

void MavlinkUDP::cleanupWinsock() {
#ifdef WIN32
    if (winsockInitialized_) {
        WSACleanup();
        winsockInitialized_ = false;
    }
#endif
}

bool MavlinkUDP::connect(const std::string& host, int port) {
    if (!initWinsock()) {
        return false;
    }

    if (connected_) {
        disconnect();
    }

#ifdef WIN32
    socket_ = ::socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (socket_ == INVALID_SOCKET) {
        return false;
    }
#else
    socket_ = ::socket(AF_INET, SOCK_DGRAM, 0);
    if (socket_ < 0) {
        return false;
    }
#endif

    targetAddr_.sin_family = AF_INET;
    targetAddr_.sin_port = htons(static_cast<uint16_t>(port));

    if (inet_pton(AF_INET, host.c_str(), &targetAddr_.sin_addr) <= 0) {
#ifdef WIN32
        closesocket(socket_);
        socket_ = INVALID_SOCKET;
#else
        ::close(socket_);
        socket_ = -1;
#endif
        return false;
    }

    struct sockaddr_in localAddr;
    std::memset(&localAddr, 0, sizeof(localAddr));
    localAddr.sin_family = AF_INET;
    localAddr.sin_addr.s_addr = htonl(INADDR_ANY);
    localAddr.sin_port = 0;

    if (::bind(socket_, reinterpret_cast<struct sockaddr*>(&localAddr), sizeof(localAddr)) != 0) {
#ifdef WIN32
        closesocket(socket_);
        socket_ = INVALID_SOCKET;
#else
        ::close(socket_);
        socket_ = -1;
#endif
        return false;
    }

#ifdef WIN32
    u_long mode = 1;
    if (ioctlsocket(socket_, FIONBIO, &mode) != 0) {
        closesocket(socket_);
        socket_ = INVALID_SOCKET;
        return false;
    }
#else
    int flags = fcntl(socket_, F_GETFL, 0);
    if (flags < 0 || fcntl(socket_, F_SETFL, flags | O_NONBLOCK) < 0) {
        ::close(socket_);
        socket_ = -1;
        return false;
    }
#endif

    connected_ = true;
    return true;
}

void MavlinkUDP::disconnect() {
    if (!connected_) return;

#ifdef WIN32
    if (socket_ != INVALID_SOCKET) {
        closesocket(socket_);
        socket_ = INVALID_SOCKET;
    }
#else
    if (socket_ >= 0) {
        ::close(socket_);
        socket_ = -1;
    }
#endif

    connected_ = false;
}

bool MavlinkUDP::isConnected() const {
    return connected_;
}

bool MavlinkUDP::sendData(const uint8_t* data, size_t length) {
    if (!connected_ || !data || length == 0) {
        return false;
    }

    ssize_t sent = ::sendto(
        socket_,
        reinterpret_cast<const char*>(data),
        static_cast<int>(length),
        0,
        reinterpret_cast<struct sockaddr*>(&targetAddr_),
        sizeof(targetAddr_)
    );

    return sent == static_cast<ssize_t>(length);
}

bool MavlinkUDP::sendData(const std::vector<uint8_t>& data) {
    return sendData(data.data(), data.size());
}

ssize_t MavlinkUDP::receiveData(uint8_t* buffer, size_t bufferSize, int timeoutMs) {
    if (!connected_ || !buffer || bufferSize == 0) {
        return -1;
    }

    fd_set readfds;
    struct timeval tv;

    FD_ZERO(&readfds);
#ifdef WIN32
    FD_SET(socket_, &readfds);
#else
    FD_SET(socket_, &readfds);
#endif

    tv.tv_sec = timeoutMs / 1000;
    tv.tv_usec = (timeoutMs % 1000) * 1000;

    int selectResult = select(
#ifdef WIN32
        0,
#else
        socket_ + 1,
#endif
        &readfds,
        nullptr,
        nullptr,
        timeoutMs > 0 ? &tv : nullptr
    );

    if (selectResult <= 0) {
        return 0;
    }

    struct sockaddr_in senderAddr;
    socklen_t senderLen = sizeof(senderAddr);

    ssize_t received = ::recvfrom(
        socket_,
        reinterpret_cast<char*>(buffer),
        static_cast<int>(bufferSize),
        0,
        reinterpret_cast<struct sockaddr*>(&senderAddr),
        &senderLen
    );

    return received;
}

std::vector<uint8_t> MavlinkUDP::receiveData(size_t maxSize, int timeoutMs) {
    std::vector<uint8_t> buffer(maxSize);
    ssize_t received = receiveData(buffer.data(), maxSize, timeoutMs);

    if (received > 0) {
        buffer.resize(static_cast<size_t>(received));
        return buffer;
    }

    return std::vector<uint8_t>();
}

void MavlinkUDP::setTargetAddress(const std::string& host, int port) {
    targetAddr_.sin_family = AF_INET;
    targetAddr_.sin_port = htons(static_cast<uint16_t>(port));
    inet_pton(AF_INET, host.c_str(), &targetAddr_.sin_addr);
}

}
