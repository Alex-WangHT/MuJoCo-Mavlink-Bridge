#pragma once

#include <vector>
#include <cstdint>
#include <chrono>
#include "mavlink_udp.h"

namespace mavlink_bridge {

struct JointControl {
    uint8_t index;
    float value;
    std::string name;

    JointControl() : index(0), value(0.0f) {}
    JointControl(uint8_t idx, float val, const std::string& n = "")
        : index(idx), value(val), name(n) {}
};

struct ControlCommand {
    uint64_t timeUsec;
    std::vector<float> controls;
    uint8_t mode;
    uint64_t flags;

    ControlCommand() : timeUsec(0), mode(0), flags(0) {
        controls.resize(16, 0.0f);
    }
};

class SimpleControl {
public:
    SimpleControl();
    ~SimpleControl();

    bool connect(const std::string& host, int port);
    void disconnect();
    bool isConnected() const;

    bool sendJointControl(uint8_t jointIndex, float value);
    bool sendMultipleJoints(const std::vector<JointControl>& controls);
    bool sendControlCommand(const ControlCommand& cmd);

    bool sendHeartbeat(uint8_t type = 2, uint8_t autopilot = 12,
                        uint8_t baseMode = 0, uint32_t customMode = 0,
                        uint8_t systemStatus = 4);

    bool sendHilStateQuaternion(
        const std::vector<float>& attitudeQuaternion,
        float rollspeed, float pitchspeed, float yawspeed,
        int32_t lat, int32_t lon, int32_t alt,
        float vx, float vy, float vz,
        float xacc, float yacc, float zacc
    );

    void setSourceSystem(uint8_t sysid) { sourceSystem_ = sysid; }
    void setSourceComponent(uint8_t compid) { sourceComponent_ = compid; }
    void setTargetSystem(uint8_t sysid) { targetSystem_ = sysid; }
    void setTargetComponent(uint8_t compid) { targetComponent_ = compid; }

private:
    MavlinkUDP udp_;
    uint8_t sourceSystem_;
    uint8_t sourceComponent_;
    uint8_t targetSystem_;
    uint8_t targetComponent_;
    uint8_t sequence_;

    uint64_t getTimestampUsec();
    uint8_t calculateChecksum(const uint8_t* data, size_t length);

    std::vector<uint8_t> createHilActuatorControlsMessage(const ControlCommand& cmd);
    std::vector<uint8_t> createHeartbeatMessage(uint8_t type, uint8_t autopilot,
                                                   uint8_t baseMode, uint32_t customMode,
                                                   uint8_t systemStatus);
};

}
