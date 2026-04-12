/*
 * ESP32 + MPU6050 IMU Streaming Firmware
 * ESP-IDF Framework — Plain C
 * ============================================================
 * Streams accelerometer data over UART at 100 Hz.
 *
 * Output (CSV over USB serial, 115200 baud):
 *   timestamp_ms,ax,ay,az
 *   1234,0.1230,-0.0510,9.7820
 *   ...
 *
 * Wiring:
 *   MPU6050 VCC  ->  ESP32 3.3V
 *   MPU6050 GND  ->  ESP32 GND
 *   MPU6050 SDA  ->  ESP32 GPIO 1       <-- changed
 *   MPU6050 SCL  ->  ESP32 GPIO 0       <-- changed
 *   MPU6050 AD0  ->  GND  (I2C addr = 0x68)
 *
 * NOTE: GPIO 0 is a boot strapping pin on classic ESP32.
 *   - It must be HIGH (or floating) at power-on for normal boot.
 *   - Keep the MPU6050 SCL line HIGH at startup; it will be
 *     pulled high by the internal I2C pull-up automatically.
 *   - If the board fails to boot, temporarily disconnect SCL
 *     while pressing reset, then reconnect.
 *   - ESP32-C3 / ESP32-S3 boards do NOT have this restriction.
 *
 * Build & Flash (ESP-IDF v5.x):
 *   idf.py set-target esp32
 *   idf.py build
 *   idf.py -p COM3 flash monitor
 * ============================================================
 */

#include <stdio.h>
#include <stdint.h>
#include <string.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/i2c.h"
#include "driver/uart.h"
#include "esp_timer.h"
#include "esp_log.h"

/* ---- Pin definitions ---------------------------------------- */
#define I2C_SDA_PIN        1          /* MPU6050 SDA */
#define I2C_SCL_PIN        0          /* MPU6050 SCL */
#define I2C_PORT           I2C_NUM_0
#define I2C_FREQ_HZ        400000     /* 400 kHz fast mode */

/* ---- MPU6050 I2C address ------------------------------------ */
#define MPU6050_ADDR       0x68       /* AD0 tied to GND */

/* ---- MPU6050 register map ----------------------------------- */
#define REG_PWR_MGMT_1     0x6B
#define REG_ACCEL_CONFIG   0x1C
#define REG_ACCEL_XOUT_H   0x3B      /* 6 bytes: XH XL YH YL ZH ZL */
#define REG_WHO_AM_I       0x75

/* ---- Accelerometer scale ------------------------------------ */
/* ACCEL_CONFIG bits [4:3] = 10 -> +/-8g -> 4096 LSB/g          */
#define ACCEL_CONFIG_8G    0x10
#define ACCEL_SCALE        4096.0f    /* LSB per g at +/-8g       */
#define G_MS2              9.80665f   /* 1g in m/s^2              */

/* ---- Sampling ------------------------------------------------ */
#define SAMPLE_RATE_HZ     100
#define SAMPLE_PERIOD_MS   (1000 / SAMPLE_RATE_HZ)   /* 10 ms   */

static const char *TAG = "imu";

/* ============================================================
 * Low-level I2C helpers
 * ============================================================ */

static esp_err_t i2c_write_reg(uint8_t reg, uint8_t val)
{
    uint8_t buf[2] = { reg, val };
    return i2c_master_write_to_device(I2C_PORT, MPU6050_ADDR,
                                      buf, sizeof(buf),
                                      pdMS_TO_TICKS(100));
}

static esp_err_t i2c_read_regs(uint8_t reg, uint8_t *out, size_t len)
{
    return i2c_master_write_read_device(I2C_PORT, MPU6050_ADDR,
                                        &reg, 1,
                                        out, len,
                                        pdMS_TO_TICKS(100));
}

/* ============================================================
 * I2C bus initialisation
 * ============================================================ */

static void i2c_init(void)
{
    i2c_config_t cfg = {
        .mode             = I2C_MODE_MASTER,
        .sda_io_num       = I2C_SDA_PIN,
        .scl_io_num       = I2C_SCL_PIN,
        .sda_pullup_en    = GPIO_PULLUP_ENABLE,
        .scl_pullup_en    = GPIO_PULLUP_ENABLE,
        .master.clk_speed = I2C_FREQ_HZ,
    };
    ESP_ERROR_CHECK(i2c_param_config(I2C_PORT, &cfg));
    ESP_ERROR_CHECK(i2c_driver_install(I2C_PORT, I2C_MODE_MASTER, 0, 0, 0));
    ESP_LOGI(TAG, "I2C initialised  SDA=GPIO%d  SCL=GPIO%d", I2C_SDA_PIN, I2C_SCL_PIN);
}

/* ============================================================
 * MPU6050 initialisation
 * ============================================================ */

static void mpu6050_init(void)
{
    /* Verify device identity */
    uint8_t who = 0;
    ESP_ERROR_CHECK(i2c_read_regs(REG_WHO_AM_I, &who, 1));
    if (who != 0x68) {
        ESP_LOGE(TAG, "MPU6050 not found! WHO_AM_I=0x%02X (expected 0x68). Check wiring.", who);
        /* Halt — Python collector will not receive data */
        while (1) vTaskDelay(pdMS_TO_TICKS(500));
    }
    ESP_LOGI(TAG, "MPU6050 detected  WHO_AM_I=0x%02X", who);

    /* Wake up: clear sleep bit in PWR_MGMT_1 */
    ESP_ERROR_CHECK(i2c_write_reg(REG_PWR_MGMT_1, 0x00));

    /* Set accelerometer range to +/-8g (best for jumps: captures ~3g takeoff) */
    ESP_ERROR_CHECK(i2c_write_reg(REG_ACCEL_CONFIG, ACCEL_CONFIG_8G));

    ESP_LOGI(TAG, "MPU6050 configured  range=+/-8g  scale=%.0f LSB/g", ACCEL_SCALE);
}

/* ============================================================
 * Read one 3-axis accelerometer sample
 * ============================================================ */

typedef struct {
    float ax;   /* m/s^2 */
    float ay;
    float az;
} accel_t;

static esp_err_t mpu6050_read_accel(accel_t *out)
{
    uint8_t raw[6];
    esp_err_t err = i2c_read_regs(REG_ACCEL_XOUT_H, raw, sizeof(raw));
    if (err != ESP_OK) return err;

    /* Registers are big-endian: H byte first */
    int16_t rx = (int16_t)((raw[0] << 8) | raw[1]);
    int16_t ry = (int16_t)((raw[2] << 8) | raw[3]);
    int16_t rz = (int16_t)((raw[4] << 8) | raw[5]);

    out->ax = (rx / ACCEL_SCALE) * G_MS2;
    out->ay = (ry / ACCEL_SCALE) * G_MS2;
    out->az = (rz / ACCEL_SCALE) * G_MS2;

    return ESP_OK;
}

/* ============================================================
 * Main streaming task
 * ============================================================ */

static void stream_task(void *arg)
{
    /* Print CSV header once — Python collector uses this to find columns */
    printf("timestamp_ms,ax,ay,az\r\n");

    TickType_t last_wake = xTaskGetTickCount();

    while (1) {
        accel_t a;
        esp_err_t err = mpu6050_read_accel(&a);

        if (err == ESP_OK) {
            uint64_t ts_ms = esp_timer_get_time() / 1000ULL;
            /*
             * printf over UART0 at 115200 baud.
             * Format matches what Python data_collector.py expects.
             */
            printf("%llu,%.4f,%.4f,%.4f\r\n",
                   (unsigned long long)ts_ms,
                   a.ax, a.ay, a.az);
        } else {
            ESP_LOGW(TAG, "I2C read error: %s", esp_err_to_name(err));
        }

        /* vTaskDelayUntil gives precise 10 ms intervals (100 Hz) */
        vTaskDelayUntil(&last_wake, pdMS_TO_TICKS(SAMPLE_PERIOD_MS));
    }
}

/* ============================================================
 * App entry point
 * ============================================================ */

void app_main(void)
{
    ESP_LOGI(TAG, "=== IMU Jumplab Firmware  v1.0 ===");
    ESP_LOGI(TAG, "Sample rate: %d Hz", SAMPLE_RATE_HZ);

    i2c_init();
    mpu6050_init();

    /* Stack size 4096 bytes is enough for printf + I2C.
     * Priority 5 keeps the sampling loop responsive. */
    xTaskCreate(stream_task, "imu_stream", 4096, NULL, 5, NULL);
}
