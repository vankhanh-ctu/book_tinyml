/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
  ==============================================================================*/

#include <TensorFlowLite.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "audio_provider.h"
#include "main_functions.h"
#include "micro_features_model.h"
#include "feature_provider.h"
#include "micro_features_micro_model_settings.h"
#include "micro_features_yes_micro_features_data.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_op_resolver.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

typedef struct Queue {
  int front, rear, capacity;
  int* queue;
  // Constructor to initialize the queue
  Queue(int c) {
    front = 0;
    rear = -1;
    capacity = c;
    queue = new int[c];
  }

  // Destructor to free the allocated memory
  ~Queue() {
    delete[] queue;
  }

  // Function to insert an element at the rear of the queue
  void queueEnqueue(int data) {
    // Check if the queue is full
    if (rear == capacity - 1) {
      printf("\nQueue is full\n");
      return;
    }

    // Insert element at the rear
    queue[++rear] = data;
  }

  // Function to delete an element from the front
  // of the queue
  void queueDequeue() {
    // If the queue is empty
    if (front > rear) {
      printf("\nQueue is empty\n");
      return;
    }

    // Shift all elements from index 1 till rear to
    // the left by one
    for (int i = 0; i < rear; i++) {
      queue[i] = queue[i + 1];
    }

    // Decrement rear
    rear--;
  }

  // Function to print queue elements
  void queueDisplay() {
    if (front > rear) {
      printf("\nQueue is Empty\n");
      return;
    }

    // Traverse front to rear and print elements
    for (int i = front; i <= rear; i++) {
      printf(" %d <-- ", queue[i]);
    }
    printf("\n");
  }
  void queue_average(float* average) {
    int sum = 0;
    if (front > rear) {
      printf("\nQueue is Empty\n");
      return;
    } else {
      for (int i = front; i <= rear; i++) {
        sum += queue[i];
      }
      *average = (float)sum / (float)(rear + 1);
    }
  }
  // Function to print the front of the queue
  void queueFront() {
    if (rear == -1) {
      printf("\nQueue is Empty\n");
      return;
    }
    printf("\nFront Element is: %d\n", queue[front]);
  }
};

namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* model_input = nullptr;
TfLiteTensor* output = nullptr;
FeatureProvider* feature_provider = nullptr;

constexpr int kTensorArenaSize = 60 * 1024;  //110 * 1024
uint8_t tensor_arena[kTensorArenaSize];
int8_t feature_buffer[kFeatureElementCount];
int8_t* model_input_buffer = nullptr;
int32_t previous_time = 0;
int16_t count[4];
}

void conduction() {

  const int32_t current_time = LatestTimes();
  int how_many_new_slices = 0;
  int t1 = micros();
  TfLiteStatus feature_status = feature_provider->PopulateFeatureData(
    error_reporter, previous_time, current_time, &how_many_new_slices);
  if (feature_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "Feature generation failed");
    return;
  }

  previous_time = current_time;
  // If no new audio samples have been received since last time, don't bother
  // running the network model.
  if (how_many_new_slices == 0) {
    return;
  }
  for (int i = 0; i < kFeatureElementCount; i++) {
    model_input_buffer[i] = feature_buffer[i];
  }

  TfLiteStatus invoke_status = interpreter->Invoke();

  if (invoke_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed");
    return;
  }


  TfLiteTensor* output = interpreter->output(0);

  // Lấy kết quả đầu ra (dạng int8)
  int8_t score_normal = output->data.int8[0];
  int8_t score_phase_loss = output->data.int8[1];
  int8_t score_phase_shift = output->data.int8[2];
  int8_t score_bearing_fail = output->data.int8[3];

  // Chuyển sang float (nếu có quantization parameters)
  // float scale = output->params.scale;
  // int zero_point = output->params.zero_point;


  float scale = 0.003921568859368563;
  int zero_point = -128;

  float f_normal = (score_normal - zero_point) * scale;
  float f_phase_loss = (score_phase_loss - zero_point) * scale;
  float f_phase_shift = (score_phase_shift - zero_point) * scale;
  float f_bearing_fail = (score_bearing_fail - zero_point) * scale;

  // In ra kết quả từng lớp
  // Serial.println("------ Model Output ------");
  // Serial.print("Normal: ");        Serial.println(f_normal, 6);
  // Serial.print("Phase Loss: ");    Serial.println(f_phase_loss, 6);
  // Serial.print("Phase Shift: ");   Serial.println(f_phase_shift, 6);
  // Serial.print("Bearing Failure: "); Serial.println(f_bearing_fail, 6);

  // Xác định lớp có xác suất cao nhất
  float scores[4] = { f_normal, f_phase_loss, f_phase_shift, f_bearing_fail };
  const char* labels[4] = { "Normal", "Phase Loss", "Phase Shift", "Bearing Failure" };

  int best_index = 0;
  float best_score = scores[0];
  for (int i = 1; i < 4; i++) {
    if (scores[i] > best_score) {
      best_score = scores[i];
      best_index = i;
    }
  }

  // Serial.print("Predicted Class: ");
  // Serial.println(labels[best_index]);
  Serial.print(best_index);
  Serial.print(", ");


  // Serial.println("--------------------------");
}


// The name of this function is important for Arduino compatibility.
void setup() {
  //setup
  Serial.begin(115200);

  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  model = tflite::GetModel(g_model);
  //  if (model->version() != TFLITE_SCHEMA_VERSION) {
  //    error_reporter->Report(
  //        "Model provided is schema version %d not equal "
  //        "to supported version %d.",
  //        model->version(), TFLITE_SCHEMA_VERSION);
  //    return;
  //   }

  static tflite::MicroMutableOpResolver<7> micro_op_resolver(error_reporter);
  if (micro_op_resolver.AddDepthwiseConv2D() != kTfLiteOk) {
    return;
  }
  if (micro_op_resolver.AddConv2D() != kTfLiteOk) {
    return;
  }
  if (micro_op_resolver.AddMaxPool2D() != kTfLiteOk) {
    return;
  }
  if (micro_op_resolver.AddAveragePool2D() != kTfLiteOk) {
    return;
  }
  if (micro_op_resolver.AddFullyConnected() != kTfLiteOk) {
    return;
  }
  if (micro_op_resolver.AddSoftmax() != kTfLiteOk) {
    return;
  }
  if (micro_op_resolver.AddReshape() != kTfLiteOk) {
    return;
  }
  static tflite::MicroInterpreter static_interpreter(
    model, micro_op_resolver, tensor_arena, kTensorArenaSize,
    error_reporter);
  interpreter = &static_interpreter;
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    error_reporter->Report("AllocateTensors() failed");
    return;
  }

  model_input = interpreter->input(0);
  model_input_buffer = model_input->data.int8;
  output = interpreter->output(0);

  static FeatureProvider static_feature_provider(kFeatureElementCount,
                                                 feature_buffer);
  feature_provider = &static_feature_provider;


  for (int i = 0; i < 420; i++) {
    conduction();
  }

}

void loop() {
}
