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


#include <driver/i2s.h>
#define RXp2 16
#define TXp2 17


// ====== SETTINGS ======
constexpr int kTensorArenaSize = 20 * 1024;


// int8_t feature_buffer[4096];
uint8_t send_buffer[4096];

bool myflag = false;
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

float output_image[64][64];  // Lưu kết quả đầu ra cho 1 ảnh (64x64)

TaskHandle_t Task1;
TaskHandle_t Task2;

SemaphoreHandle_t xSerialSemaphore;

// ====== HÀM DEQUANTIZE ======
inline float DequantizeInt8(int8_t value, float scale, int zero_point) {
  return (static_cast<float>(value) - static_cast<float>(zero_point)) * scale;
}

// ====== HÀM CONDUCTION: CHẠY 1 ẢNH 64x64 ======
void conduction_image() {
  const int32_t current_time = LatestTimes();
  int how_many_new_slices = 0;

  // Tạo feature từ audio hoặc dữ liệu nguồn
  TfLiteStatus feature_status = feature_provider->PopulateFeatureData(
    error_reporter, previous_time, current_time, &how_many_new_slices);
  if (feature_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "Feature generation failed");
    return;
  }
  previous_time = current_time;
  if (how_many_new_slices == 0) return;


  for (int i = 0; i < 4096; i++) {
    send_buffer[i] = (feature_buffer[i]);

  }

  while (Serial2.read() != 0x31)
    ;
  Serial2.write(send_buffer, sizeof(send_buffer));

  vTaskDelay(pdMS_TO_TICKS(1));
}

void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial2.begin(256000, SERIAL_8N1, RXp2, TXp2);
  xSerialSemaphore = xSemaphoreCreateMutex();
  Serial.println("1");



  Serial.println("=== TensorFlow Lite Micro - 64x64 Image Reconstruction ===");

  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Load model
  model = tflite::GetModel(g_model);
  // if (model->version() != TFLITE_SCHEMA_VERSION) {
  //   error_reporter->Report("Model schema version mismatch");
  //   return;
  // }

  // Register ops
  static tflite::MicroMutableOpResolver<7> micro_op_resolver(error_reporter);
  micro_op_resolver.AddDepthwiseConv2D();
  micro_op_resolver.AddConv2D();
  micro_op_resolver.AddMaxPool2D();
  micro_op_resolver.AddAveragePool2D();
  micro_op_resolver.AddFullyConnected();
  micro_op_resolver.AddSoftmax();
  micro_op_resolver.AddReshape();

  // Create interpreter
  static tflite::MicroInterpreter static_interpreter(
    model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate tensors
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

void loop() {
  conduction_image();
}

void Task1code(void* pvParameters) {
  Serial.print("Task1 running on core ");
  Serial.println(xPortGetCoreID());
  for (;;) {

    myflag = true;

    vTaskDelay(pdMS_TO_TICKS(100));
  }
}
