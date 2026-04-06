/*
 * ESP32 + MPU6050 IMU Streaming Firmware
 * ----------------------------------------
 * Streams raw accelerometer data over Serial at 100 Hz.
 * Output format (CSV line): timestamp_ms,ax,ay,az
 * Units: acceleration in m/s^2, timestamp in milliseconds
 *
 * Wiring (I2C):
 *   MPU6050 VCC  -> ESP32 3.3V
 *   MPU6050 GND  -> ESP32 GND
 *   MPU6050 SCL  -> ESP32 GPIO 22
 *   MPU6050 SDA  -> ESP32 GPIO 21
 *   MPU6050 AD0  -> GND (I2C address = 0x68)
 *
 * Dependencies (Arduino Library Manager):
 *   - Adafruit MPU6050
 *   - Adafruit Unified Sensor
 */

#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Wire.h>

Adafruit_MPU6050 mpu;

const int SAMPLE_RATE_HZ = 100;
const int SAMPLE_INTERVAL_MS = 1000 / SAMPLE_RATE_HZ;  // 10 ms

unsigned long lastSampleTime = 0;

void setup() {
  Serial.begin(115200);
  while (!Serial) delay(10);

  if (!mpu.begin()) {
    Serial.println("ERROR: MPU6050 not found. Check wiring.");
    while (1) delay(100);
  }

  // Configure accelerometer range: +/- 8g is a good range for jumps
  mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
  mpu.setGyroRange(MPU6050_RANGE_500_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);

  // Print CSV header so Python knows column names
  Serial.println("timestamp_ms,ax,ay,az");

  delay(100);
}

void loop() {
  unsigned long now = millis();

  if (now - lastSampleTime >= SAMPLE_INTERVAL_MS) {
    lastSampleTime = now;

    sensors_event_t accel, gyro, temp;
    mpu.getEvent(&accel, &gyro, &temp);

    // Stream: timestamp, ax, ay, az (m/s^2)
    Serial.print(now);
    Serial.print(",");
    Serial.print(accel.acceleration.x, 4);
    Serial.print(",");
    Serial.print(accel.acceleration.y, 4);
    Serial.print(",");
    Serial.println(accel.acceleration.z, 4);
  }
}
