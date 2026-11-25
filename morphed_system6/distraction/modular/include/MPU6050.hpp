#pragma once

#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>

#if defined(__linux__)
#include <fcntl.h>
#include <linux/i2c-dev.h>
#include <sys/ioctl.h>
#include <unistd.h>
#endif

/**
 * Data structures for MPU-6050 readings
 */
struct AccelData {
    float x{0.0f};   // m/s^2
    float y{0.0f};   // m/s^2
    float z{0.0f};   // m/s^2
    float mag{0.0f}; // total magnitude sqrt(x^2 + y^2 + z^2)
};

struct GyroData {
    float x{0.0f};   // deg/s
    float y{0.0f};   // deg/s
    float z{0.0f};   // deg/s
};

struct IMUReading {
    AccelData accel{};
    GyroData  gyro{};
    float     temperature{0.0f}; // Celsius
    bool      valid{false};
};

class MPU6050 {
public:
    static constexpr uint8_t DEFAULT_ADDRESS = 0x68;
    static constexpr float   GRAVITY_CONST   = 9.80665f;
    static constexpr float   ACCEL_SCALE_2G  = 16384.0f; // LSB/g for +/-2g
    static constexpr float   GYRO_SCALE_250  = 131.0f;   // LSB/(deg/s) for +/-250 deg/s

    // Register map
    static constexpr uint8_t REG_SMPLRT_DIV   = 0x19;
    static constexpr uint8_t REG_CONFIG       = 0x1A;
    static constexpr uint8_t REG_GYRO_CONFIG  = 0x1B;
    static constexpr uint8_t REG_ACCEL_CONFIG = 0x1C;
    static constexpr uint8_t REG_ACCEL_XOUT_H = 0x3B;
    static constexpr uint8_t REG_PWR_MGMT_1   = 0x6B;
    static constexpr uint8_t REG_WHO_AM_I     = 0x75;

#if defined(__linux__)
    MPU6050(const std::string& i2c_device = "/dev/i2c-1", uint8_t address = DEFAULT_ADDRESS)
        : m_i2c_device(i2c_device), m_address(address), m_file_descriptor(-1) {}

    ~MPU6050() {
        close_connection();
    }

    /**
     * Open I2C bus and wake sensor up
     */
    bool init() {
        m_file_descriptor = open(m_i2c_device.c_str(), O_RDWR);
        if (m_file_descriptor < 0) {
            std::cerr << "[MPU6050] Error: Failed to open I2C bus: " << m_i2c_device << std::endl;
            return false;
        }

        if (ioctl(m_file_descriptor, I2C_SLAVE, m_address) < 0) {
            std::cerr << "[MPU6050] Error: Could not acquire bus access to slave 0x"
                      << std::hex << (int)m_address << std::dec << std::endl;
            close_connection();
            return false;
        }

        // Verify WHO_AM_I register (should return 0x68)
        uint8_t who_am_i = read_byte(REG_WHO_AM_I);
        if (who_am_i != 0x68) {
            std::cerr << "[MPU6050] Warning: WHO_AM_I returned 0x"
                      << std::hex << (int)who_am_i << " (expected 0x68)" << std::dec << std::endl;
        }

        // Wake up MPU-6050 by clearing sleep bit (bit 6 of PWR_MGMT_1)
        if (!write_byte(REG_PWR_MGMT_1, 0x00)) {
            std::cerr << "[MPU6050] Error: Failed to wake up sensor." << std::endl;
            close_connection();
            return false;
        }

        // Optional: Set sample rate divider (e.g. 1kHz / (1 + 7) = 125Hz)
        write_byte(REG_SMPLRT_DIV, 0x07);

        // Low-pass DLPF setting: 44Hz bandwidth (good for car vibration filtering)
        write_byte(REG_CONFIG, 0x03);

        // Accelerometer config: +/- 2g
        write_byte(REG_ACCEL_CONFIG, 0x00);

        // Gyroscope config: +/- 250 deg/s
        write_byte(REG_GYRO_CONFIG, 0x00);

        return true;
    }

    /**
     * Read all 14 sensor registers in one burst
     */
    IMUReading read() {
        IMUReading reading{};
        reading.valid = false;

        if (m_file_descriptor < 0) {
            return reading;
        }

        uint8_t buffer[14];
        if (!read_bytes(REG_ACCEL_XOUT_H, buffer, 14)) {
            return reading;
        }

        // Combine high and low bytes (Big-endian signed 16-bit)
        int16_t raw_ax = (int16_t)((buffer[0] << 8) | buffer[1]);
        int16_t raw_ay = (int16_t)((buffer[2] << 8) | buffer[3]);
        int16_t raw_az = (int16_t)((buffer[4] << 8) | buffer[5]);

        int16_t raw_temp = (int16_t)((buffer[6] << 8) | buffer[7]);

        int16_t raw_gx = (int16_t)((buffer[8]  << 8) | buffer[9]);
        int16_t raw_gy = (int16_t)((buffer[10] << 8) | buffer[11]);
        int16_t raw_gz = (int16_t)((buffer[12] << 8) | buffer[13]);

        // Convert to physical units
        reading.accel.x = (raw_ax / ACCEL_SCALE_2G) * GRAVITY_CONST;
        reading.accel.y = (raw_ay / ACCEL_SCALE_2G) * GRAVITY_CONST;
        reading.accel.z = (raw_az / ACCEL_SCALE_2G) * GRAVITY_CONST;
        reading.accel.mag = std::sqrt(
            reading.accel.x * reading.accel.x +
            reading.accel.y * reading.accel.y +
            reading.accel.z * reading.accel.z
        );

        reading.gyro.x = raw_gx / GYRO_SCALE_250;
        reading.gyro.y = raw_gy / GYRO_SCALE_250;
        reading.gyro.z = raw_gz / GYRO_SCALE_250;

        reading.temperature = (raw_temp / 340.0f) + 36.53f;
        reading.valid = true;

        return reading;
    }

    void close_connection() {
        if (m_file_descriptor >= 0) {
            close(m_file_descriptor);
            m_file_descriptor = -1;
        }
    }

private:
    std::string m_i2c_device;
    uint8_t     m_address;
    int         m_file_descriptor;

    bool write_byte(uint8_t reg, uint8_t value) {
        uint8_t buf[2] = {reg, value};
        return (::write(m_file_descriptor, buf, 2) == 2);
    }

    uint8_t read_byte(uint8_t reg) {
        if (::write(m_file_descriptor, &reg, 1) != 1) return 0;
        uint8_t val = 0;
        if (::read(m_file_descriptor, &val, 1) != 1) return 0;
        return val;
    }

    bool read_bytes(uint8_t reg, uint8_t* buffer, size_t count) {
        if (::write(m_file_descriptor, &reg, 1) != 1) {
            return false;
        }
        return (::read(m_file_descriptor, buffer, count) == (ssize_t)count);
    }
#else
    MPU6050(const std::string& i2c_device = "/dev/i2c-1", uint8_t address = DEFAULT_ADDRESS) {
        (void)i2c_device;
        (void)address;
    }

    ~MPU6050() = default;

    bool init() {
#ifdef DEBUG_VIEWS
        std::cout << "[MPU6050] Linux I2C unavailable on this platform; hardware IMU disabled.\n";
#endif
        return false;
    }

    IMUReading read() {
        return IMUReading{};
    }

    void close_connection() {}
#endif
};
