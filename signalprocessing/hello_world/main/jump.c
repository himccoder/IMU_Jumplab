#include <stdio.h>
#include <math.h>
#include <stdbool.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "driver/i2c_master.h"
#include "esp_log.h"

#define I2C_SDA_PIN           8
#define I2C_SCL_PIN           9
#define I2C_PORT              I2C_NUM_0

#define MPU6050_ADDR          0x68
#define MPU6050_PWR_MGMT_1    0x6B
#define MPU6050_ACCEL_XOUT_H  0x3B

#define FREEFALL_THRESHOLD    0.35f
#define LANDING_THRESHOLD     1.80f
#define MIN_AIRTIME_MS        120
#define MAX_AIRTIME_MS        1000

static const char *TAG = "JUMP_COUNTER";

i2c_master_bus_handle_t bus_handle;
i2c_master_dev_handle_t mpu_handle;

int jump_count = 0;
bool airborne = false;
uint32_t jump_start_time = 0;

void i2c_init()
{
    i2c_master_bus_config_t bus_config = {
        .clk_source = I2C_CLK_SRC_DEFAULT,
        .i2c_port = I2C_PORT,
        .sda_io_num = I2C_SDA_PIN,
        .scl_io_num = I2C_SCL_PIN,
        .glitch_ignore_cnt = 7,
        .flags.enable_internal_pullup = true,
    };

    ESP_ERROR_CHECK(i2c_new_master_bus(&bus_config, &bus_handle));

    i2c_device_config_t dev_cfg = {
        .dev_addr_length = I2C_ADDR_BIT_LEN_7,
        .device_address = MPU6050_ADDR,
        .scl_speed_hz = 400000,
    };

    ESP_ERROR_CHECK(i2c_master_bus_add_device(bus_handle, &dev_cfg, &mpu_handle));
}

void mpu6050_write(uint8_t reg, uint8_t data)
{
    uint8_t buffer[2] = {reg, data};
    ESP_ERROR_CHECK(i2c_master_transmit(mpu_handle, buffer, 2, -1));
}

void mpu6050_read(uint8_t reg, uint8_t *data, size_t len)
{
    ESP_ERROR_CHECK(
        i2c_master_transmit_receive(
            mpu_handle,
            &reg,
            1,
            data,
            len,
            -1
        )
    );
}

void mpu6050_init()
{
    mpu6050_write(MPU6050_PWR_MGMT_1, 0x00);
    ESP_LOGI(TAG, "MPU6050 initialized");
}

void detect_jump(float magnitude)
{
    uint32_t now = xTaskGetTickCount() * portTICK_PERIOD_MS;

    if (!airborne && magnitude < FREEFALL_THRESHOLD)
    {
        airborne = true;
        jump_start_time = now;
        ESP_LOGI(TAG, "Airborne detected");
    }

    if (airborne && magnitude > LANDING_THRESHOLD)
    {
        uint32_t airtime_ms = now - jump_start_time;

        airborne = false;

        if (airtime_ms >= MIN_AIRTIME_MS && airtime_ms <= MAX_AIRTIME_MS)
        {
            jump_count++;

            float airtime_s = airtime_ms / 1000.0f;

            // h = g * t^2 / 8
            float jump_height_m = (9.81f * airtime_s * airtime_s) / 8.0f;
            float jump_height_cm = jump_height_m * 100.0f;

            ESP_LOGI(
                TAG,
                "Jump %d | Airtime: %lu ms | Height: %.1f cm",
                jump_count,
                airtime_ms,
                jump_height_cm
            );
        }
    }
}

void app_main(void)
{
    ESP_LOGI(TAG, "Starting basic jump counter");

    i2c_init();
    mpu6050_init();

    uint8_t raw_data[6];

    while (1)
    {
        mpu6050_read(MPU6050_ACCEL_XOUT_H, raw_data, 6);

        int16_t ax = (raw_data[0] << 8) | raw_data[1];
        int16_t ay = (raw_data[2] << 8) | raw_data[3];
        int16_t az = (raw_data[4] << 8) | raw_data[5];

        float x = ax / 16384.0f;
        float y = ay / 16384.0f;
        float z = az / 16384.0f;

        float magnitude = sqrtf((x * x) + (y * y) + (z * z));

        detect_jump(magnitude);

        ESP_LOGI(
            TAG,
            "Accel magnitude: %.2f g | Jumps: %d",
            magnitude,
            jump_count
        );

        vTaskDelay(pdMS_TO_TICKS(20));
    }
}