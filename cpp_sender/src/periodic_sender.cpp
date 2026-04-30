#include <iostream>
#include <thread>
#include <chrono>
#include <vector>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <iomanip>
#include <atomic>
#include <csignal>

#include "mavlink_udp.h"

using namespace mavlink_bridge;

namespace {
    std::atomic<bool> gRunning(true);
}

void signalHandler(int signal) {
    gRunning = false;
}

struct ControlConfig {
    float amplitude = 1.0f;
    float frequency = 0.5f;
    float phase = 0.0f;
    float offset = 0.0f;
    std::string waveType = "sin";
};

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

float generateWave(const ControlConfig& config, double time) {
    double t = time * 2.0 * M_PI * config.frequency + config.phase;

    if (config.waveType == "sin") {
        return config.offset + config.amplitude * static_cast<float>(std::sin(t));
    } else if (config.waveType == "cos") {
        return config.offset + config.amplitude * static_cast<float>(std::cos(t));
    } else if (config.waveType == "square") {
        float val = std::sin(t) >= 0 ? 1.0f : -1.0f;
        return config.offset + config.amplitude * val;
    } else if (config.waveType == "triangle") {
        double normalized = std::fmod(t, 2.0 * M_PI);
        double val;
        if (normalized < M_PI) {
            val = 2.0 * normalized / M_PI - 1.0;
        } else {
            val = 3.0 - 2.0 * normalized / M_PI;
        }
        return config.offset + config.amplitude * static_cast<float>(val);
    } else if (config.waveType == "sawtooth") {
        double normalized = std::fmod(t, 2.0 * M_PI);
        double val = normalized / M_PI - 1.0;
        return config.offset + config.amplitude * static_cast<float>(val);
    }

    return config.offset;
}

void printUsage() {
    std::cout << "Periodic MAVLink Sender\n";
    std::cout << "\nUsage: periodic_sender [options]\n";
    std::cout << "\nOptions:\n";
    std::cout << "  --host <ip>       Target IP (default: 127.0.0.1)\n";
    std::cout << "  --port <num>      Target port (default: 14540)\n";
    std::cout << "  --rate <hz>       Send rate (default: 50)\n";
    std::cout << "  --joints <n>      Number of joints to control (default: 4)\n";
    std::cout << "  --amplitude <a>   Wave amplitude (default: 1.0)\n";
    std::cout << "  --frequency <f>   Wave frequency in Hz (default: 0.5)\n";
    std::cout << "  --wave <type>     Wave type: sin, cos, square, triangle, sawtooth (default: sin)\n";
    std::cout << "  --help            Show this help\n";
}

int main(int argc, char* argv[]) {
    std::string host = "127.0.0.1";
    int port = 14540;
    int rate = 50;
    int numJoints = 4;
    float amplitude = 1.0f;
    float frequency = 0.5f;
    std::string waveType = "sin";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help") {
            printUsage();
            return 0;
        } else if (arg == "--host" && i + 1 < argc) {
            host = argv[++i];
        } else if (arg == "--port" && i + 1 < argc) {
            port = std::stoi(argv[++i]);
        } else if (arg == "--rate" && i + 1 < argc) {
            rate = std::stoi(argv[++i]);
        } else if (arg == "--joints" && i + 1 < argc) {
            numJoints = std::stoi(argv[++i]);
        } else if (arg == "--amplitude" && i + 1 < argc) {
            amplitude = std::stof(argv[++i]);
        } else if (arg == "--frequency" && i + 1 < argc) {
            frequency = std::stof(argv[++i]);
        } else if (arg == "--wave" && i + 1 < argc) {
            waveType = argv[++i];
        }
    }

    std::signal(SIGINT, signalHandler);
#ifdef SIGTERM
    std::signal(SIGTERM, signalHandler);
#endif

    std::cout << "========================================\n";
    std::cout << "Periodic MAVLink Sender\n";
    std::cout << "========================================\n";
    std::cout << "Target: " << host << ":" << port << "\n";
    std::cout << "Rate: " << rate << " Hz\n";
    std::cout << "Joints: " << numJoints << "\n";
    std::cout << "Wave: " << waveType << "\n";
    std::cout << "Amplitude: " << amplitude << "\n";
    std::cout << "Frequency: " << frequency << " Hz\n";
    std::cout << "========================================\n";

    std::vector<ControlConfig> jointConfigs;
    for (int i = 0; i < numJoints; ++i) {
        ControlConfig cfg;
        cfg.amplitude = amplitude;
        cfg.frequency = frequency;
        cfg.phase = static_cast<float>(i) * 0.5f;
        cfg.offset = 0.0f;
        cfg.waveType = waveType;
        jointConfigs.push_back(cfg);
    }

    MavlinkUDP udp;
    if (!udp.connect(host, port)) {
        std::cerr << "Failed to connect to " << host << ":" << port << "\n";
        return 1;
    }

    std::cout << "Connected. Press Ctrl+C to stop.\n\n";

    auto startTime = std::chrono::high_resolution_clock::now();
    auto lastPrintTime = startTime;
    uint64_t messageCount = 0;

    while (gRunning) {
        auto now = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(now - startTime).count();
        double t = static_cast<double>(elapsed) / 1000000.0;
        uint64_t timeUsec = static_cast<uint64_t>(elapsed);

        std::vector<float> controls(16, 0.0f);
        for (int i = 0; i < numJoints && i < 16; ++i) {
            controls[i] = generateWave(jointConfigs[i], t);
        }

        auto msg = createHilActuatorControlsMessage(timeUsec, controls, 0, 0);

        if (udp.sendData(msg)) {
            messageCount++;
        } else {
            std::cerr << "Send failed\n";
        }

        auto printElapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastPrintTime).count();
        if (printElapsed >= 1000) {
            std::cout << "Time: " << std::fixed << std::setprecision(2) << t << "s | ";
            for (int i = 0; i < numJoints && i < 4; ++i) {
                std::cout << "J" << i << ": " << std::fixed << std::setprecision(3) << controls[i] << " | ";
            }
            std::cout << "Total: " << messageCount << "\n";
            lastPrintTime = now;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(1000 / rate));
    }

    std::cout << "\nStopping...\n";
    udp.disconnect();
    std::cout << "Total messages sent: " << messageCount << "\n";

    return 0;
}
