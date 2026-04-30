#include "simple_control.h"
#include <cstring>
#include <cmath>

namespace mavlink_bridge {

SimpleControl::SimpleControl()
    : sourceSystem_(1)
    , sourceComponent_(1)
    , targetSystem_(1)
    , targetComponent_(1)
    , sequence_(0)
{}

SimpleControl::~SimpleControl() {
    disconnect();
}

bool SimpleControl::connect(const std::string& host, int port) {
    return udp_.connect(host, port);
}

void SimpleControl::disconnect() {
    udp_.disconnect();
}

bool SimpleControl::isConnected() const {
    return udp_.isConnected();
}

uint64_t SimpleControl::getTimestampUsec() {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
}

uint8_t SimpleControl::calculateChecksum(const uint8_t* data, size_t length) {
    uint8_t checksum = 0;
    for (size_t i = 0; i < length; ++i) {
        checksum ^= data[i];
    }
    return checksum;
}

std::vector<uint8_t> SimpleControl::createHilActuatorControlsMessage(const ControlCommand& cmd) {
    std::vector<uint8_t> message;

    message.push_back(0xFD);

    uint8_t flags = 0;
    uint8_t incompatFlags = 0;
    uint8_t compatFlags = 0;

    message.push_back(64);
    message.push_back(incompatFlags);
    message.push_back(compatFlags);

    message.push_back(sequence_++);

    message.push_back(sourceSystem_);
    message.push_back(sourceComponent_);

    uint32_t msgId = 93;
    message.push_back(static_cast<uint8_t>(msgId & 0xFF));
    message.push_back(static_cast<uint8_t>((msgId >> 8) & 0xFF));
    message.push_back(static_cast<uint8_t>((msgId >> 16) & 0xFF));

    uint64_t timeUsec = cmd.timeUsec > 0 ? cmd.timeUsec : getTimestampUsec();
    for (int i = 0; i < 8; ++i) {
        message.push_back(static_cast<uint8_t>((timeUsec >> (i * 8)) & 0xFF));
    }

    for (int i = 0; i < 16; ++i) {
        float value = i < cmd.controls.size() ? cmd.controls[i] : 0.0f;
        uint32_t floatBits;
        std::memcpy(&floatBits, &value, sizeof(float));
        for (int j = 0; j < 4; ++j) {
            message.push_back(static_cast<uint8_t>((floatBits >> (j * 8)) & 0xFF));
        }
    }

    message.push_back(cmd.mode);

    uint64_t flags64 = cmd.flags;
    for (int i = 0; i < 8; ++i) {
        message.push_back(static_cast<uint8_t>((flags64 >> (i * 8)) & 0xFF));
    }

    uint16_t checksum = 0xFFFF;
    for (size_t i = 1; i < message.size(); ++i) {
        uint8_t tmp = message[i] ^ (checksum & 0xFF);
        tmp ^= (tmp << 4);
        checksum = (checksum >> 8) ^ (tmp << 8) ^ (tmp << 3) ^ (tmp >> 4);
    }

    message.push_back(static_cast<uint8_t>(checksum & 0xFF));
    message.push_back(static_cast<uint8_t>((checksum >> 8) & 0xFF));

    return message;
}

std::vector<uint8_t> SimpleControl::createHeartbeatMessage(uint8_t type, uint8_t autopilot,
                                                              uint8_t baseMode, uint32_t customMode,
                                                              uint8_t systemStatus) {
    std::vector<uint8_t> message;

    message.push_back(0xFD);

    uint8_t incompatFlags = 0;
    uint8_t compatFlags = 0;

    message.push_back(9);
    message.push_back(incompatFlags);
    message.push_back(compatFlags);

    message.push_back(sequence_++);

    message.push_back(sourceSystem_);
    message.push_back(sourceComponent_);

    uint32_t msgId = 0;
    message.push_back(static_cast<uint8_t>(msgId & 0xFF));
    message.push_back(static_cast<uint8_t>((msgId >> 8) & 0xFF));
    message.push_back(static_cast<uint8_t>((msgId >> 16) & 0xFF));

    for (int i = 0; i < 4; ++i) {
        message.push_back(static_cast<uint8_t>((customMode >> (i * 8)) & 0xFF));
    }

    message.push_back(type);
    message.push_back(autopilot);
    message.push_back(baseMode);
    message.push_back(systemStatus);
    message.push_back(3);

    uint16_t checksum = 0xFFFF;
    for (size_t i = 1; i < message.size(); ++i) {
        uint8_t tmp = message[i] ^ (checksum & 0xFF);
        tmp ^= (tmp << 4);
        checksum = (checksum >> 8) ^ (tmp << 8) ^ (tmp << 3) ^ (tmp >> 4);
    }

    message.push_back(static_cast<uint8_t>(checksum & 0xFF));
    message.push_back(static_cast<uint8_t>((checksum >> 8) & 0xFF));

    return message;
}

bool SimpleControl::sendJointControl(uint8_t jointIndex, float value) {
    ControlCommand cmd;
    if (jointIndex < 16) {
        cmd.controls[jointIndex] = value;
    }
    return sendControlCommand(cmd);
}

bool SimpleControl::sendMultipleJoints(const std::vector<JointControl>& controls) {
    ControlCommand cmd;
    for (const auto& jc : controls) {
        if (jc.index < 16) {
            cmd.controls[jc.index] = jc.value;
        }
    }
    return sendControlCommand(cmd);
}

bool SimpleControl::sendControlCommand(const ControlCommand& cmd) {
    if (!udp_.isConnected()) {
        return false;
    }

    std::vector<uint8_t> message = createHilActuatorControlsMessage(cmd);
    return udp_.sendData(message);
}

bool SimpleControl::sendHeartbeat(uint8_t type, uint8_t autopilot,
                                    uint8_t baseMode, uint32_t customMode,
                                    uint8_t systemStatus) {
    if (!udp_.isConnected()) {
        return false;
    }

    std::vector<uint8_t> message = createHeartbeatMessage(type, autopilot, baseMode, customMode, systemStatus);
    return udp_.sendData(message);
}

bool SimpleControl::sendHilStateQuaternion(
    const std::vector<float>& attitudeQuaternion,
    float rollspeed, float pitchspeed, float yawspeed,
    int32_t lat, int32_t lon, int32_t alt,
    float vx, float vy, float vz,
    float xacc, float yacc, float zacc
) {
    if (!udp_.isConnected()) {
        return false;
    }

    std::vector<uint8_t> message;

    message.push_back(0xFD);

    message.push_back(56);
    message.push_back(0);
    message.push_back(0);

    message.push_back(sequence_++);

    message.push_back(sourceSystem_);
    message.push_back(sourceComponent_);

    uint32_t msgId = 115;
    message.push_back(static_cast<uint8_t>(msgId & 0xFF));
    message.push_back(static_cast<uint8_t>((msgId >> 8) & 0xFF));
    message.push_back(static_cast<uint8_t>((msgId >> 16) & 0xFF));

    uint64_t timeUsec = getTimestampUsec();
    for (int i = 0; i < 8; ++i) {
        message.push_back(static_cast<uint8_t>((timeUsec >> (i * 8)) & 0xFF));
    }

    for (int i = 0; i < 4; ++i) {
        float q = i < attitudeQuaternion.size() ? attitudeQuaternion[i] : 0.0f;
        if (i == 0 && attitudeQuaternion.empty()) q = 1.0f;
        uint32_t floatBits;
        std::memcpy(&floatBits, &q, sizeof(float));
        for (int j = 0; j < 4; ++j) {
            message.push_back(static_cast<uint8_t>((floatBits >> (j * 8)) & 0xFF));
        }
    }

    float speeds[] = {rollspeed, pitchspeed, yawspeed};
    for (int i = 0; i < 3; ++i) {
        uint32_t floatBits;
        std::memcpy(&floatBits, &speeds[i], sizeof(float));
        for (int j = 0; j < 4; ++j) {
            message.push_back(static_cast<uint8_t>((floatBits >> (j * 8)) & 0xFF));
        }
    }

    int32_t gps[] = {lat, lon, alt};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 4; ++j) {
            message.push_back(static_cast<uint8_t>((gps[i] >> (j * 8)) & 0xFF));
        }
    }

    float vel[] = {vx, vy, vz};
    for (int i = 0; i < 3; ++i) {
        uint32_t floatBits;
        std::memcpy(&floatBits, &vel[i], sizeof(float));
        for (int j = 0; j < 4; ++j) {
            message.push_back(static_cast<uint8_t>((floatBits >> (j * 8)) & 0xFF));
        }
    }

    float acc[] = {xacc, yacc, zacc};
    for (int i = 0; i < 3; ++i) {
        uint32_t floatBits;
        std::memcpy(&floatBits, &acc[i], sizeof(float));
        for (int j = 0; j < 4; ++j) {
            message.push_back(static_cast<uint8_t>((floatBits >> (j * 8)) & 0xFF));
        }
    }

    uint16_t checksum = 0xFFFF;
    for (size_t i = 1; i < message.size(); ++i) {
        uint8_t tmp = message[i] ^ (checksum & 0xFF);
        tmp ^= (tmp << 4);
        checksum = (checksum >> 8) ^ (tmp << 8) ^ (tmp << 3) ^ (tmp >> 4);
    }

    message.push_back(static_cast<uint8_t>(checksum & 0xFF));
    message.push_back(static_cast<uint8_t>((checksum >> 8) & 0xFF));

    return udp_.sendData(message);
}

}
