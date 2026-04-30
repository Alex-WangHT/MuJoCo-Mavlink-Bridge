#include <iostream>
#include <thread>
#include <chrono>
#include <iomanip>
#include <csignal>
#include "simple_control.h"

using namespace mavlink_bridge;

namespace {
    volatile std::sig_atomic_t gRunning = 1;
}

void signalHandler(int signal) {
    gRunning = 0;
}

void printHelp() {
    std::cout << "MAVLink Sender - Control MuJoCo simulation via MAVLink\n";
    std::cout << "\nUsage:\n";
    std::cout << "  mavlink_sender [options]\n";
    std::cout << "\nOptions:\n";
    std::cout << "  --host <ip>       Target IP address (default: 127.0.0.1)\n";
    std::cout << "  --port <num>      Target port (default: 14540)\n";
    std::cout << "  --joint <idx>     Joint index to control (0-15, default: 0)\n";
    std::cout << "  --value <val>     Control value (default: 0.0)\n";
    std::cout << "  --mode <type>     Control mode: sin, step, ramp, manual (default: sin)\n";
    std::cout << "  --rate <hz>       Send rate in Hz (default: 50)\n";
    std::cout << "  --amplitude <a>   Amplitude for sine mode (default: 1.0)\n";
    std::cout << "  --frequency <f>   Frequency for sine mode in Hz (default: 0.5)\n";
    std::cout << "  --multi <n>       Number of joints to control (default: 1)\n";
    std::cout << "  --help            Show this help message\n";
    std::cout << "\nExamples:\n";
    std::cout << "  mavlink_sender --mode sin --amplitude 2.0 --frequency 1.0\n";
    std::cout << "  mavlink_sender --joint 0 --value 1.5 --mode step\n";
    std::cout << "  mavlink_sender --multi 4 --mode sin\n";
}

int main(int argc, char* argv[]) {
    std::string host = "127.0.0.1";
    int port = 14540;
    int jointIndex = 0;
    float value = 0.0f;
    std::string mode = "sin";
    int rate = 50;
    float amplitude = 1.0f;
    float frequency = 0.5f;
    int multiJoints = 1;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            printHelp();
            return 0;
        } else if (arg == "--host" && i + 1 < argc) {
            host = argv[++i];
        } else if (arg == "--port" && i + 1 < argc) {
            port = std::stoi(argv[++i]);
        } else if (arg == "--joint" && i + 1 < argc) {
            jointIndex = std::stoi(argv[++i]);
        } else if (arg == "--value" && i + 1 < argc) {
            value = std::stof(argv[++i]);
        } else if (arg == "--mode" && i + 1 < argc) {
            mode = argv[++i];
        } else if (arg == "--rate" && i + 1 < argc) {
            rate = std::stoi(argv[++i]);
        } else if (arg == "--amplitude" && i + 1 < argc) {
            amplitude = std::stof(argv[++i]);
        } else if (arg == "--frequency" && i + 1 < argc) {
            frequency = std::stof(argv[++i]);
        } else if (arg == "--multi" && i + 1 < argc) {
            multiJoints = std::stoi(argv[++i]);
        }
    }

    std::signal(SIGINT, signalHandler);
#ifdef SIGTERM
    std::signal(SIGTERM, signalHandler);
#endif

    std::cout << "========================================\n";
    std::cout << "MAVLink Sender\n";
    std::cout << "========================================\n";
    std::cout << "Target: " << host << ":" << port << "\n";
    std::cout << "Mode: " << mode << "\n";
    std::cout << "Rate: " << rate << " Hz\n";
    if (mode == "sin") {
        std::cout << "Amplitude: " << amplitude << "\n";
        std::cout << "Frequency: " << frequency << " Hz\n";
    }
    std::cout << "========================================\n";

    SimpleControl control;

    if (!control.connect(host, port)) {
        std::cerr << "Error: Failed to connect to " << host << ":" << port << "\n";
        return 1;
    }

    std::cout << "Connected. Press Ctrl+C to stop.\n\n";

    auto startTime = std::chrono::high_resolution_clock::now();
    auto lastPrintTime = startTime;
    int messageCount = 0;

    while (gRunning) {
        auto now = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(now - startTime).count();
        double t = static_cast<double>(elapsed) / 1000000.0;

        std::vector<JointControl> controls;

        if (mode == "sin") {
            for (int i = 0; i < multiJoints; ++i) {
                float phase = static_cast<float>(i) * 0.5f;
                float val = amplitude * std::sin(2.0f * static_cast<float>(M_PI) * frequency * static_cast<float>(t) + phase);
                controls.emplace_back(static_cast<uint8_t>(jointIndex + i), val);
            }
        } else if (mode == "step") {
            float val = value;
            for (int i = 0; i < multiJoints; ++i) {
                controls.emplace_back(static_cast<uint8_t>(jointIndex + i), val * (i + 1) / static_cast<float>(multiJoints));
            }
        } else if (mode == "ramp") {
            float val = static_cast<float>(std::fmod(t * 0.5, 2.0) - 1.0) * amplitude;
            for (int i = 0; i < multiJoints; ++i) {
                controls.emplace_back(static_cast<uint8_t>(jointIndex + i), val * (i + 1) / static_cast<float>(multiJoints));
            }
        } else {
            controls.emplace_back(static_cast<uint8_t>(jointIndex), value);
        }

        if (control.sendMultipleJoints(controls)) {
            messageCount++;
        } else {
            std::cerr << "Warning: Failed to send message\n";
        }

        if (messageCount % 50 == 0) {
            auto printElapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastPrintTime).count();
            if (printElapsed >= 1000) {
                std::cout << "Time: " << std::fixed << std::setprecision(2) << t << "s | ";
                for (const auto& c : controls) {
                    std::cout << "J" << static_cast<int>(c.index) << ": " 
                              << std::fixed << std::setprecision(3) << c.value << " | ";
                }
                std::cout << "Msgs: " << messageCount << "\n";
                lastPrintTime = now;
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(1000 / rate));
    }

    std::cout << "\nStopping...\n";
    control.disconnect();
    std::cout << "Total messages sent: " << messageCount << "\n";

    return 0;
}
