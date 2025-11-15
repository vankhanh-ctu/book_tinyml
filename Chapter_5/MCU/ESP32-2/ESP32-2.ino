#include <SPI.h>
#include <Ethernet.h>
#include "MgsModbus.h"
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <TensorFlowLite.h>
#include <math.h>
#include "audio_provider.h"
#include "main_functions.h"
#include "micro_features_model.h"
#include "feature_provider.h"
#include "micro_features_micro_model_settings.h"
#include "micro_features_yes_micro_features_data.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"



#define RXp2 16
#define TXp2 17
#define ETH_CS 5
#define ETH_SCLK 18
#define ETH_MISO 19
#define ETH_MOSI 23


// ====== SETTINGS ======
constexpr int kTensorArenaSize = 20 * 1024;
// =======================

namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* model_input = nullptr;
TfLiteTensor* output = nullptr;
FeatureProvider* feature_provider = nullptr;

uint8_t tensor_arena[kTensorArenaSize];
int8_t feature_buffer[kFeatureElementCount];
int8_t* model_input_buffer = nullptr;
int32_t previous_time = 0;
}  // namespace

float output_image[64][64];
char buffer1[4096] = {};
float mse;

// ====== DEQUANTIZE FUNCTION ======
inline float DequantizeInt8(int8_t value, float scale, int zero_point) {
  return (static_cast<float>(value) - static_cast<float>(zero_point)) * scale;
}

TaskHandle_t Task1;

// ====== MODBUS VARIABLES ======
MgsModbus Mb;
byte mac[] = { 0x90, 0xA2, 0xDA, 0x0E, 0x94, 0xB5 };
IPAddress ip(192, 168, 0, 120);
IPAddress gateway(192, 168, 0, 2);
IPAddress subnet(255, 255, 255, 0);

// ====== IMAGE INFERENCE FUNCTION ======
void conduction_image() {
  Serial2.write("1");
  while (!Serial2.available())
    ;
  Serial2.readBytes(buffer1, 4096);

  const int32_t current_time = LatestTimes();
  int how_many_new_slices = 0;

  TfLiteStatus feature_status = feature_provider->PopulateFeatureData(
    error_reporter, previous_time, current_time, &how_many_new_slices);
  if (feature_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "Feature generation failed");
    return;
  }

  previous_time = current_time;
  if (how_many_new_slices == 0) return;

  const float in_scale = 0.0037216455675661564;
  const int in_zero_point = -128;
  const float out_scale = 0.0036092763766646385;
  const int out_zero_point = -128;


  // if (Mb.MbData[2]==true) {
    for (int i = 0; i < 64; i++) {
      for (int j = 0; j < 64; j++) {
        model_input_buffer[j] = int8_t(buffer1[i * 64 + j]);
      }

      if (interpreter->Invoke() != kTfLiteOk) {
        TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed at row %d", i);
        return;
      }

      int8_t* y_quantized = output->data.int8;
      for (int k = 0; k < 64; k++) {
        float pred = DequantizeInt8(y_quantized[k], out_scale, out_zero_point);
        output_image[i][k] = pred;
      }
    }

    float total_err = 0.0f;
    for (int i = 0; i < 64; i++) {
      for (int j = 0; j < 64; j++) {
        float in_f = DequantizeInt8(buffer1[i * 64 + j], in_scale, in_zero_point);
        float out_f = output_image[i][j];
        float diff = in_f - out_f;
        total_err += diff * diff;
      }
    }

    mse = (total_err / (64.0f * 64.0f)) * 100000;
  // }
  Serial.printf("%.8f\n", mse);
}

// ====== SETUP ======
void setup() {
  Serial.begin(115200);
  delay(1000);

  // Create Modbus Task
  xTaskCreatePinnedToCore(
    Task1code,
    "Task1",
    10000,
    NULL,
    2,
    &Task1,
    0);

  // Initialize Ethernet
  SPI.begin(ETH_SCLK, ETH_MISO, ETH_MOSI);
  Ethernet.init(ETH_CS);
  Ethernet.begin(mac, ip, gateway, subnet);
  Serial.print("IP Address: ");
  Serial.println(Ethernet.localIP());

  // Initialize Modbus registers
  Mb.MbData[0] = 0;
  Mb.MbData[1] = 0;
  Mb.MbData[2] = 0;
  Mb.MbData[3] = 0;
  Mb.MbData[4] = 0;

  Serial2.begin(256000, SERIAL_8N1, RXp2, TXp2);
  pinMode(26, OUTPUT);
  pinMode(27, OUTPUT);
  pinMode(14, OUTPUT);
  pinMode(12, OUTPUT);
  pinMode(0, INPUT_PULLUP);

  Serial.println("=== TensorFlow Lite Micro - 64x64 Image Reconstruction ===");

  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  model = tflite::GetModel(g_model);

  static tflite::MicroMutableOpResolver<7> micro_op_resolver(error_reporter);
  micro_op_resolver.AddDepthwiseConv2D();
  micro_op_resolver.AddConv2D();
  micro_op_resolver.AddMaxPool2D();
  micro_op_resolver.AddAveragePool2D();
  micro_op_resolver.AddFullyConnected();
  micro_op_resolver.AddSoftmax();
  micro_op_resolver.AddReshape();

  static tflite::MicroInterpreter static_interpreter(
    model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    error_reporter->Report("AllocateTensors() failed");
    return;
  }

  model_input = interpreter->input(0);
  model_input_buffer = model_input->data.int8;
  output = interpreter->output(0);

  static FeatureProvider static_feature_provider(kFeatureElementCount, feature_buffer);
  feature_provider = &static_feature_provider;
}
// ====== LOOP ======
void loop() {
  conduction_image();
}

// ====== MODBUS TASK ======
void Task1code(void* pvParameters) {
  int count = 0;
  for (;;) {
    Mb.MbData[0] = count++;
    Mb.MbData[1] = (int)mse;
    digitalWrite(2, Mb.MbData[2]);
    // Serial.print("40000: " + String(Mb.MbData[0]));
    // Serial.print("\n40001: " + String(Mb.MbData[1]));
    Mb.MbsRun();
    vTaskDelay(pdMS_TO_TICKS(1000));
  }
}
