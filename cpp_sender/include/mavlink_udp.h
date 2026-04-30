#pragma once

#include <string>
#include <vector>
#include <cstdint>

#ifdef WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#endif

namespace mavlink_bridge {

class MavlinkUDP {
public:
    MavlinkUDP();
    ~MavlinkUDP();

    bool connect(const std::string& host, int port);
    void disconnect();
    bool isConnected() const;

    bool sendData(const uint8_t* data, size_t length);
    bool sendData(const std::vector<uint8_t>& data);

    ssize_t receiveData(uint8_t* buffer, size_t bufferSize, int timeoutMs = 100);
    std::vector<uint8_t> receiveData(size_t maxSize, int timeoutMs = 100);

    void setTargetAddress(const std::string& host, int port);

private:
#ifdef WIN32
    SOCKET socket_;
    WSADATA wsaData_;
#else
    int socket_;
#endif

    struct sockaddr_in targetAddr_;
    bool connected_;
    bool winsockInitialized_;

    bool initWinsock();
    void cleanupWinsock();
};

}
