/*
    ============================================================
    ESP32-S3 + MPU6050
    Activity Classifier + Step Counter + Jump Detector
    ============================================================

    FEATURES
    ------------------------------------------------------------
    - Detect STILL
    - Detect WALKING
    - Detect RUNNING
    - Detect JUMPING

    - Count steps
    - Count jumps
    - Estimate jump height

    ============================================================
    CONNECTIONS
    ------------------------------------------------------------

    MPU6050      ESP32-S3
    --------     ----------
    VCC       -> 3.3V
    GND       -> GND
    SDA       -> GPIO 8
    SCL       -> GPIO 9

    ============================================================
    HOW IT WORKS
    ------------------------------------------------------------

    1. Read accelerometer
    2. Compute acceleration magnitude
    3. Remove gravity
    4. Filter motion signal
    5. Detect peaks for steps
    6. Detect freefall + landing for jumps
    7. Use step timing to classify:
            WALKING vs RUNNING

    ============================================================
*/

#include <stdio.h>
#include <math.h>
#include <stdbool.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "driver/i2c_master.h"

#include "esp_log.h"

// ============================================================
// I2C CONFIGURATION
// ============================================================

#define I2C_SDA_PIN           8
#define I2C_SCL_PIN           9

#define I2C_PORT              I2C_NUM_0

#define MPU6050_ADDR          0x68

// MPU6050 Registers
#define MPU6050_PWR_MGMT_1    0x6B
#define MPU6050_ACCEL_XOUT_H  0x3B

static const char *TAG = "FITNESS_TRACKER";

// ============================================================
// ACTIVITY STATES
// ============================================================

typedef enum
{
    STATE_STILL,
    STATE_WALKING,
    STATE_RUNNING,
    STATE_JUMPING

} motion_state_t;

motion_state_t current_state = STATE_STILL;

// ============================================================
// GLOBAL VARIABLES
// ============================================================

// Counters
int total_steps = 0;
int total_jumps = 0;

// Step detection
bool step_detected = false;

uint32_t last_step_time = 0;
uint32_t previous_step_time = 0;

// Jump detection
bool airborne = false;

uint32_t jump_start_time = 0;

// Filter
float filtered_motion = 0;

// ============================================================
// TUNING PARAMETERS
// ============================================================

// Step threshold
#define STEP_THRESHOLD        0.18

// Reset threshold
#define STEP_RESET_THRESHOLD  0.05

// Minimum delay between steps
#define STEP_DELAY_MS         250

// Low-pass filter
#define FILTER_ALPHA          0.90

// Still threshold
#define STILL_THRESHOLD       0.03

// Jump thresholds
#define FREEFALL_THRESHOLD    0.30
#define LANDING_THRESHOLD     2.20

// ============================================================
// I2C HANDLES
// ============================================================

i2c_master_bus_handle_t bus_handle;
i2c_master_dev_handle_t mpu_handle;

// ============================================================
// I2C INITIALIZATION
// ============================================================

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

    ESP_ERROR_CHECK(
        i2c_new_master_bus(
            &bus_config,
            &bus_handle
        )
    );

    i2c_device_config_t dev_cfg = {
        .dev_addr_length = I2C_ADDR_BIT_LEN_7,
        .device_address = MPU6050_ADDR,
        .scl_speed_hz = 400000,
    };

    ESP_ERROR_CHECK(
        i2c_master_bus_add_device(
            bus_handle,
            &dev_cfg,
            &mpu_handle
        )
    );
}

// ============================================================
// WRITE MPU6050 REGISTER
// ============================================================

void mpu6050_write(uint8_t reg, uint8_t data)
{
    uint8_t buffer[2] = {reg, data};

    ESP_ERROR_CHECK(
        i2c_master_transmit(
            mpu_handle,
            buffer,
            2,
            -1
        )
    );
}

// ============================================================
// READ MPU6050 REGISTERS
// ============================================================

void mpu6050_read(
    uint8_t reg,
    uint8_t *data,
    size_t len
)
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

// ============================================================
// MPU6050 INITIALIZATION
// ============================================================

void mpu6050_init()
{
    // Wake MPU6050
    mpu6050_write(
        MPU6050_PWR_MGMT_1,
        0x00
    );

    ESP_LOGI(TAG, "MPU6050 Initialized");
}

// ============================================================
// PRINT CURRENT STATE
// ============================================================

const char* get_state_name(motion_state_t state)
{
    switch(state)
    {
        case STATE_STILL:
            return "STILL";

        case STATE_WALKING:
            return "WALKING";

        case STATE_RUNNING:
            return "RUNNING";

        case STATE_JUMPING:
            return "JUMPING";

        default:
            return "UNKNOWN";
    }
}

// ============================================================
// STEP DETECTION
// ============================================================

void detect_step(float motion)
{
    uint32_t now =
        xTaskGetTickCount() *
        portTICK_PERIOD_MS;

    // Rising edge
    if (motion > STEP_THRESHOLD &&
        !step_detected)
    {
        // Debounce
        if ((now - last_step_time)
            > STEP_DELAY_MS)
        {
            total_steps++;

            // Time between steps
            uint32_t interval =
                now - previous_step_time;

            previous_step_time =
                last_step_time;

            last_step_time = now;

            // =================================================
            // CLASSIFY WALKING VS RUNNING
            // =================================================

            if(interval > 400)
            {
                current_state =
                    STATE_WALKING;
            }
            else
            {
                current_state =
                    STATE_RUNNING;
            }

            ESP_LOGI(
                TAG,
                "STEP %d | %s",
                total_steps,
                get_state_name(current_state)
            );
        }

        step_detected = true;
    }

    // Reset step state
    if (motion < STEP_RESET_THRESHOLD)
    {
        step_detected = false;
    }
}

// ============================================================
// JUMP DETECTION
// ============================================================

void detect_jump(float magnitude)
{
    uint32_t now =
        xTaskGetTickCount() *
        portTICK_PERIOD_MS;

    // =========================================================
    // FREEFALL DETECTION
    // =========================================================
    //
    // During jump airtime,
    // acceleration approaches 0g.
    //

    if(magnitude < FREEFALL_THRESHOLD &&
       !airborne)
    {
        airborne = true;

        jump_start_time = now;
    }

    // =========================================================
    // LANDING DETECTION
    // =========================================================
    //
    // Landing creates large acceleration spike.
    //

    if(airborne &&
       magnitude > LANDING_THRESHOLD)
    {
        airborne = false;

        total_jumps++;

        current_state =
            STATE_JUMPING;

        // =====================================================
        // ESTIMATE JUMP HEIGHT
        // =====================================================

        uint32_t airtime_ms =
            now - jump_start_time;

        float airtime_s =
            airtime_ms / 1000.0;

        // Physics:
        // h = g * t² / 8

        float jump_height =
            (9.81 * airtime_s * airtime_s)
            / 8.0;

        // Convert to centimeters
        jump_height *= 100.0;

        ESP_LOGI(
            TAG,
            "JUMP %d | Height: %.1f cm",
            total_jumps,
            jump_height
        );
    }
}

// ============================================================
// STILL DETECTION
// ============================================================

void detect_still(float motion)
{
    static uint32_t still_timer = 0;

    uint32_t now =
        xTaskGetTickCount() *
        portTICK_PERIOD_MS;

    if(fabs(motion) < STILL_THRESHOLD)
    {
        if(still_timer == 0)
        {
            still_timer = now;
        }

        // Still for >2 seconds
        if((now - still_timer) > 2000)
        {
            current_state =
                STATE_STILL;
        }
    }
    else
    {
        still_timer = 0;
    }
}

// ============================================================
// MAIN APPLICATION
// ============================================================

void app_main(void)
{
    ESP_LOGI(
        TAG,
        "Starting Activity Tracker"
    );

    // Initialize I2C
    i2c_init();

    // Initialize MPU6050
    mpu6050_init();

    uint8_t raw_data[6];

    while (1)
    {
        // =====================================================
        // READ ACCELEROMETER
        // =====================================================

        mpu6050_read(
            MPU6050_ACCEL_XOUT_H,
            raw_data,
            6
        );

        // Combine high + low bytes
        int16_t ax =
            (raw_data[0] << 8)
            | raw_data[1];

        int16_t ay =
            (raw_data[2] << 8)
            | raw_data[3];

        int16_t az =
            (raw_data[4] << 8)
            | raw_data[5];

        // =====================================================
        // CONVERT TO g
        // =====================================================

        float x = ax / 16384.0;
        float y = ay / 16384.0;
        float z = az / 16384.0;

        // =====================================================
        // MAGNITUDE
        // =====================================================

        float magnitude =
            sqrtf(
                x*x +
                y*y +
                z*z
            );

        // Remove gravity
        float motion =
            magnitude - 1.0;

        // =====================================================
        // FILTER
        // =====================================================

        filtered_motion =
            FILTER_ALPHA *
            filtered_motion +

            (1.0 - FILTER_ALPHA) *
            motion;

        // =====================================================
        // DETECT ACTIVITIES
        // =====================================================

        detect_step(filtered_motion);

        detect_jump(magnitude);

        detect_still(filtered_motion);

        // =====================================================
        // DEBUG OUTPUT
        // =====================================================

        ESP_LOGI(
            TAG,
            "State:%s | Motion:%.2f | Steps:%d | Jumps:%d",
            get_state_name(current_state),
            filtered_motion,
            total_steps,
            total_jumps
        );

        // 50Hz sample rate
        vTaskDelay(pdMS_TO_TICKS(20));
    }
}