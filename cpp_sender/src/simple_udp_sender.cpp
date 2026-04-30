#include <iostream>
#include <thread>
#include <chrono>
#include <vector>
#include <cstdint>
#include <cstring>
#include <cmath>

#ifdef WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#endif

class SimpleUDPSender {
public:
    SimpleUDPSender()
#ifdef WIN32
        : socket_(INVALID_SOCKET)
#else
        : socket_(-1)
#endif
    {}

    ~SimpleUDPSender() {
        close();
    }

    bool init(const std::string& host, int port) {
#ifdef WIN32
        WSADATA wsaData;
        if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
            std::cerr << "WSAStartup failed\n";
            return false;
        }
#endif

        socket_ = socket(AF_INET, SOCK_DGRAM, 0);
#ifdef WIN32
        if (socket_ == INVALID_SOCKET) {
#else
        if (socket_ < 0) {
#endif
            std::cerr << "Failed to create socket\n";
            return false;
        }

        std::memset(&targetAddr_, 0, sizeof(targetAddr_));
        targetAddr_.sin_family = AF_INET;
        targetAddr_.sin_port = htons(static_cast<uint16_t>(port));

        if (inet_pton(AF_INET, host.c_str(), &targetAddr_.sin_addr) <= 0) {
            std::cerr << "Invalid address: " << host << "\n";
            return false;
        }

        return true;
    }

    void close() {
#ifdef WIN32
        if (socket_ != INVALID_SOCKET) {
            closesocket(socket_);
            socket_ = INVALID_SOCKET;
        }
        WSACleanup();
#else
        if (socket_ >= 0) {
            ::close(socket_);
            socket_ = -1;
        }
#endif
    }

    bool send(const std::vector<uint8_t>& data) {
        ssize_t sent = sendto(
            socket_,
            reinterpret_cast<const char*>(data.data()),
            static_cast<int>(data.size()),
            0,
            reinterpret_cast<struct sockaddr*>(&targetAddr_),
            sizeof(targetAddr_)
        );
        return sent == static_cast<ssize_t>(data.size());
    }

    std::vector<uint8_t> createHilActuatorControlsMessage(
        uint64_t timeUsec,
        const std::vector<float>& controls,
        uint8_t mode,
        uint64_t flags
    ) {
        std::vector<uint8_t> msg;

        msg.push_back(0xFD);

        msg.push_back(64);
        msg.push_back(0);
        msg.push_back(0);

        static uint8_t seq = 0;
        msg.push_back(seq++);

        msg.push_back(1);
        msg.push_back(1);

        uint32_t msgId = 93;
        msg.push_back(static_cast<uint8_t>(msgId & 0xFF));
        msg.push_back(static_cast<uint8_t>((msgId >> 8) & 0xFF));
        msg.push_back(static_cast<uint8_t>((msgId >> 16) & 0xFF));

        for (int i = 0; i < 8; ++i) {
            msg.push_back(static_cast<uint8_t>((timeUsec >> (i * 8)) & 0xFF));
        }

        for (int i = 0; i < 16; ++i) {
            float val = i < controls.size() ? controls[i] : 0.0f;
            uint32_t floatBits;
            std::memcpy(&floatBits, &val, sizeof(float));
            for (int j = 0; j < 4; ++j) {
                msg.push_back(static_cast<uint8_t>((floatBits >> (j * 8)) & 0xFF));
            }
        }

        msg.push_back(mode);

        for (int i = 0; i < 8; ++i) {
            msg.push_back(static_cast<uint8_t>((flags >> (i * 8)) & 0xFF));
        }

        uint16_t checksum = 0xFFFF;
        for (size_t i = 1; i < msg.size(); ++i) {
            uint8_t tmp = msg[i] ^ (checksum & 0xFF);
            tmp ^= (tmp << 4);
            checksum = (checksum >> 8) ^ (tmp << 8) ^ (tmp << 3) ^ (tmp >> 4);
        }

        msg.push_back(static_cast<uint8_t>(checksum & 0xFF));
        msg.push_back(static_cast<uint8_t>((checksum >> 8) & 0xFF));

        return msg;
    }

private:
#ifdef WIN32
    SOCKET socket_;
#else
    int socket_;
#endif
    struct sockaddr_in targetAddr_;
};

int main(int argc, char* argv[]) {
    std::string host = "127.0.0.1";
    int port = 14540;
    int rate = 50;
    float amplitude = 1.0f;
    float frequency = 0.5f;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--host" && i + 1 < argc) host = argv[++i];
        else if (arg == "--port" && i + 1 < argc) port = std::stoi(argv[++i]);
        else if (arg == "--rate" && i + 1 < argc) rate = std::stoi(argv[++i]);
        else if (arg == "--amplitude" && i + 1 < argc) amplitude = std::stof(argv[++i]);
        else if (arg == "--frequency" && i + 1 < argc) frequency = std::stof(argv[++i]);
    }

    std::cout << "Simple UDP Sender\n";
    std::cout << "Target: " << host << ":" << port << "\n";
    std::cout << "Rate: " << rate << " Hz\n\n";

    SimpleUDPSender sender;
    if (!sender.init(host, port)) {
        std::cerr << "Failed to initialize sender\n";
        return 1;
    }

    auto startTime = std::chrono::high_resolution_clock::now();
    int count = 0;

    std::cout << "Sending... Press Ctrl+C to stop\n";

    while (true) {
        auto now = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(now - startTime).count();
        double t = static_cast<double>(elapsed) / 1000000.0;
        uint64_t timeUsec = static_cast<uint64_t>(elapsed);

        std::vector<float> controls(16, 0.0f);
        controls[0] = amplitude * static_cast<float>(std::sin(2.0 * M_PI * frequency * t));
        controls[1] = amplitude * static_cast<float>(std::cos(2.0 * M_PI * frequency * t));

        auto msg = sender.createHilActuatorControlsMessage(timeUsec, controls, 0, 0);

        if (sender.send(msg)) {
            count++;
        } else {
            std::cerr << "Send failed\n";
        }

        if (count % 50 == 0) {
            std::cout << "Time: " << std::fixed << std::setprecision(2) << t 
                      << "s | Control0: " << std::fixed << std::setprecision(3) << controls[0]
                      << " | Control1: " << std::fixed << std::setprecision(3) << controls[1]
                      << " | Total: " << count << "\n";
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(1000 / rate));
    }

    sender.close();
    return 0;
}
