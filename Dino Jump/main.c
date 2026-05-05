#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>

#include "driver/i2c.h"
#include "esp_err.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "mpu6050.h"

// I2C wiring: SDA -> GPIO0, SCL -> GPIO1
#define I2C_MASTER_SDA_IO   0
#define I2C_MASTER_SCL_IO   1
#define I2C_MASTER_NUM      I2C_NUM_0
#define I2C_MASTER_FREQ_HZ  100000

// Sampling period: 100 Hz
#define SAMPLE_PERIOD_MS    10

static const char *TAG = "DINO_IMU";
static mpu6050_handle_t g_mpu = NULL;

/*
 * Tune these values for your setup.
 *
 * LPF_ALPHA:
 *   Higher = smoother gravity estimate, less sensitive to tilt.
 *
 * JUMP_THRESHOLD_G:
 *   Higher = harder to trigger jump.
 *
 * REARM_THRESHOLD_G:
 *   Motion must settle below this before another jump is allowed.
 *
 * JUMP_COOLDOWN_MS:
 *   Minimum time between jumps.
 */
#define LPF_ALPHA            0.90f
#define JUMP_THRESHOLD_G     0.35f
#define REARM_THRESHOLD_G    0.10f
#define JUMP_COOLDOWN_MS     400

// Choose the MPU6050 axis that points "up/down" in your physical setup.
// Try 'z', 'y', or 'x' if detection is not behaving as expected.
typedef enum {
    AXIS_X = 0,
    AXIS_Y,
    AXIS_Z
} motion_axis_t;

#define MOTION_AXIS AXIS_Z

static void i2c_master_init(void)
{
    i2c_config_t conf = {
        .mode = I2C_MODE_MASTER,
        .sda_io_num = I2C_MASTER_SDA_IO,
        .scl_io_num = I2C_MASTER_SCL_IO,
        .sda_pullup_en = GPIO_PULLUP_ENABLE,
        .scl_pullup_en = GPIO_PULLUP_ENABLE,
        .master.clk_speed = I2C_MASTER_FREQ_HZ,
        .clk_flags = 0,
    };

    ESP_ERROR_CHECK(i2c_param_config(I2C_MASTER_NUM, &conf));
    ESP_ERROR_CHECK(i2c_driver_install(I2C_MASTER_NUM, conf.mode, 0, 0, 0));
}

static void mpu6050_init_sensor(void)
{
    g_mpu = mpu6050_create(I2C_MASTER_NUM, MPU6050_I2C_ADDRESS);
    if (g_mpu == NULL) {
        ESP_LOGE(TAG, "mpu6050_create() failed (NULL). Check wiring/address.");
        return;
    }

    ESP_ERROR_CHECK(mpu6050_config(g_mpu, ACCE_FS_4G, GYRO_FS_500DPS));
    ESP_ERROR_CHECK(mpu6050_wake_up(g_mpu));

    ESP_LOGI(TAG, "MPU6050 initialized at I2C addr 0x%02X", MPU6050_I2C_ADDRESS);
}

static float get_motion_axis_value(const mpu6050_acce_value_t *accel)
{
#if MOTION_AXIS == AXIS_X
    return accel->acce_x;
#elif MOTION_AXIS == AXIS_Y
    return accel->acce_y;
#else
    return accel->acce_z;
#endif
}

static void trigger_jump(void)
{
    // This is what your Python script should listen for.
    printf("JUMP\n");
    fflush(stdout);
}

void app_main(void)
{
    mpu6050_acce_value_t accel;
    mpu6050_gyro_value_t gyro;

    i2c_master_init();
    mpu6050_init_sensor();

    if (g_mpu == NULL) {
        ESP_LOGE(TAG, "Sensor init failed; halting.");
        while (1) {
            vTaskDelay(pdMS_TO_TICKS(1000));
        }
    }

    const TickType_t period = pdMS_TO_TICKS(SAMPLE_PERIOD_MS);
    TickType_t last_wake = xTaskGetTickCount();

    float gravity_est = 0.0f;
    bool gravity_initialized = false;

    bool armed = true;
    int64_t last_jump_ms = 0;

    ESP_LOGI(TAG, "Started. Watching for upward motion...");

    while (1) {
        if (mpu6050_get_acce(g_mpu, &accel) == ESP_OK &&
            mpu6050_get_gyro(g_mpu, &gyro) == ESP_OK)
        {
            float motion_axis = get_motion_axis_value(&accel);

            // Initialize the gravity estimate with the first sample.
            if (!gravity_initialized) {
                gravity_est = motion_axis;
                gravity_initialized = true;
            }

            // Low-pass filter for gravity / slow tilt.
            gravity_est = LPF_ALPHA * gravity_est + (1.0f - LPF_ALPHA) * motion_axis;

            // Dynamic acceleration removes gravity and slow movement.
            float dyn_accel = motion_axis - gravity_est;

            int64_t now_ms = xTaskGetTickCount() * portTICK_PERIOD_MS;

            // Rearm only after movement settles.
            if (fabsf(dyn_accel) < REARM_THRESHOLD_G) {
                armed = true;
            }

            // Detect a quick upward jerk.
            if (armed &&
                dyn_accel > JUMP_THRESHOLD_G &&
                (now_ms - last_jump_ms) > JUMP_COOLDOWN_MS)
            {
                trigger_jump();
                last_jump_ms = now_ms;
                armed = false;
            }

            // Uncomment for tuning in serial monitor:
            // printf("raw=%.3f grav=%.3f dyn=%.3f armed=%d\n",
            //        motion_axis, gravity_est, dyn_accel, armed ? 1 : 0);
            // fflush(stdout);
        }

        vTaskDelayUntil(&last_wake, period);
    }
}