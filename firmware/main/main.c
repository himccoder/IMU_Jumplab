/*
 * ESP32 + MPU6050 IMU Streaming Firmware
 * ESP-IDF v5.x — New I2C Master Driver
 * ============================================================
 * Streams accelerometer data over UART0 at 100 Hz.
 *
 * Output (CSV over USB serial, 115200 baud):
 *   timestamp_ms,ax,ay,az
 *   1234,0.1230,-0.0510,9.7820
 *   ...
 *
 * Wiring:
 *   MPU6050 VCC  ->  ESP32 3.3V
 *   MPU6050 GND  ->  ESP32 GND
 *   MPU6050 SDA  ->  ESP32 GPIO 18
 *   MPU6050 SCL  ->  ESP32 GPIO 19
 *   MPU6050 AD0  ->  GND  (I2C addr = 0x68)
 *
 * Build & Flash (ESP-IDF v5.x):
 *   idf.py set-target esp32
 *   idf.py build
 *   idf.py -p COM6 flash monitor
 * ============================================================
 */

#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include <string.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/i2c_master.h"   /* New I2C master API — ESP-IDF v5.1+ */
#include "esp_timer.h"
#include "esp_log.h"

/* ---- Pin definitions ---------------------------------------- */
#define I2C_SDA_PIN        18         /* MPU6050 SDA -> ESP32 GPIO 18 */
#define I2C_SCL_PIN        19         /* MPU6050 SCL -> ESP32 GPIO 19 */
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
#define G_MS2              9.80665f   /* 1 g in m/s²              */

/* ---- Sampling ------------------------------------------------ */
#define SAMPLE_RATE_HZ     100
#define SAMPLE_PERIOD_MS   (1000 / SAMPLE_RATE_HZ)   /* 10 ms   */

static const char *TAG = "imu";

/* I2C device handle — shared between init and streaming task */
static i2c_master_dev_handle_t s_dev_handle;

/* ============================================================
 * Low-level I2C helpers (new API)
 * ============================================================ */

static esp_err_t i2c_write_reg(uint8_t reg, uint8_t val)
{
    uint8_t buf[2] = { reg, val };
    return i2c_master_transmit(s_dev_handle, buf, sizeof(buf), pdMS_TO_TICKS(100));
}

static esp_err_t i2c_read_regs(uint8_t reg, uint8_t *out, size_t len)
{
    return i2c_master_transmit_receive(
        s_dev_handle,
        &reg, 1,
        out, len,
        pdMS_TO_TICKS(100)
    );
}

/* ============================================================
 * I2C bus + device initialisation (new API)
 * ============================================================ */

static void i2c_init(void)
{
    i2c_master_bus_config_t bus_cfg = {
        .i2c_port            = I2C_PORT,
        .sda_io_num          = I2C_SDA_PIN,
        .scl_io_num          = I2C_SCL_PIN,
        .clk_source          = I2C_CLK_SRC_DEFAULT,
        .glitch_ignore_cnt   = 7,
        .flags.enable_internal_pullup = true,
    };
    i2c_master_bus_handle_t bus_handle;
    ESP_ERROR_CHECK(i2c_new_master_bus(&bus_cfg, &bus_handle));

    i2c_device_config_t dev_cfg = {
        .dev_addr_length = I2C_ADDR_BIT_LEN_7,
        .device_address  = MPU6050_ADDR,
        .scl_speed_hz    = I2C_FREQ_HZ,
    };
    ESP_ERROR_CHECK(i2c_master_bus_add_device(bus_handle, &dev_cfg, &s_dev_handle));

    ESP_LOGI(TAG, "I2C initialised  SDA=GPIO%d  SCL=GPIO%d", I2C_SDA_PIN, I2C_SCL_PIN);
}

/* ============================================================
 * MPU6050 initialisation
 * ============================================================ */

static void mpu6050_init(void)
{
    uint8_t who = 0;
    ESP_ERROR_CHECK(i2c_read_regs(REG_WHO_AM_I, &who, 1));
    if (who != 0x68) {
        ESP_LOGE(TAG, "MPU6050 not found! WHO_AM_I=0x%02X (expected 0x68). Check wiring.", who);
        while (1) vTaskDelay(pdMS_TO_TICKS(500));
    }
    ESP_LOGI(TAG, "MPU6050 detected  WHO_AM_I=0x%02X", who);

    /* Wake the sensor (clear sleep bit) */
    ESP_ERROR_CHECK(i2c_write_reg(REG_PWR_MGMT_1, 0x00));

    /* Set accelerometer to ±8g — captures the ~3g takeoff spike in jumps */
    ESP_ERROR_CHECK(i2c_write_reg(REG_ACCEL_CONFIG, ACCEL_CONFIG_8G));

    ESP_LOGI(TAG, "MPU6050 configured  range=+/-8g  scale=%.0f LSB/g", ACCEL_SCALE);
}

/* ============================================================
 * Read one 3-axis accelerometer sample
 * ============================================================ */

typedef struct {
    float ax;   /* m/s² */
    float ay;
    float az;
} accel_t;

static esp_err_t mpu6050_read_accel(accel_t *out)
{
    uint8_t raw[6];
    esp_err_t err = i2c_read_regs(REG_ACCEL_XOUT_H, raw, sizeof(raw));
    if (err != ESP_OK) return err;

    /* Registers are big-endian: high byte first */
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
    /* Print CSV header once — tells Python collector which columns are which */
    printf("timestamp_ms,ax,ay,az\r\n");

    TickType_t last_wake = xTaskGetTickCount();

    while (1) {
        accel_t a;
        esp_err_t err = mpu6050_read_accel(&a);

        if (err == ESP_OK) {
            uint64_t ts_ms = esp_timer_get_time() / 1000ULL;
            /* PRIu64 from <inttypes.h> gives the correct format on all targets */
            printf("%" PRIu64 ",%.4f,%.4f,%.4f\r\n", ts_ms, a.ax, a.ay, a.az);
        } else {
            ESP_LOGW(TAG, "I2C read error: %s", esp_err_to_name(err));
        }

        /* vTaskDelayUntil gives precise 10 ms periods (100 Hz) */
        vTaskDelayUntil(&last_wake, pdMS_TO_TICKS(SAMPLE_PERIOD_MS));
    }
}

/* ============================================================
 * App entry point
 * ============================================================ */

void app_main(void)
{
    ESP_LOGI(TAG, "=== IMU Jumplab Firmware  v1.1 ===");
    ESP_LOGI(TAG, "Sample rate: %d Hz", SAMPLE_RATE_HZ);

    i2c_init();
    mpu6050_init();

    xTaskCreate(stream_task, "imu_stream", 4096, NULL, 5, NULL);
}
