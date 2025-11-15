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

#ifndef TENSORFLOW_LITE_MICRO_EXAMPLES_MICRO_SPEECH_MICRO_FEATURES_MICRO_MODEL_SETTINGS_H_
#define TENSORFLOW_LITE_MICRO_EXAMPLES_MICRO_SPEECH_MICRO_FEATURES_MICRO_MODEL_SETTINGS_H_

// Keeping these as constant expressions allow us to allocate fixed-sized arrays
// on the stack for our working memory.

// The size of the input time series data we pass to the FFT to produce the
// frequency information. This has to be a power of two, and since we're dealing
// with 30ms of 16KHz inputs, which means 480 samples, this is the next value.

constexpr int kAudioSampleFrequency = 16000;

// The following values are derived from values used during model training.
// If you change the way you preprocess the input, update all these constants.
constexpr int kFeatureSliceCount = 64; // số của sổ thời gian aka số hàng của ảnh
constexpr int kFeatureSliceSize = 256; // số khoản tần số aka số cột của ảnh
constexpr int kFeatureElementCount = (kFeatureSliceSize * kFeatureSliceCount);
constexpr int kFeatureSliceStrideMs = 15; // khoảng trược của cửa sổ dữ liệu (ms)
constexpr int kFeatureSliceDurationMs = 55; // độ rộng của cửa sổ dữ liệu (ms)

// Công thức liên hệ giữa thời gian tín hiệu, số cửa sổ thời gian, độ rộng cửa sổ trược và khoảng cách trược:
//                           độ dài đoạn dữ liệu (ms) - kFeatureSliceDurationMs (ms)                      
//  kFeatureSliceCount =  ----------------------------------------------------------- + 1     
//                                               kFeatureSliceStrideMs (ms)
// Giá trị kFeatureSliceSizen tùy chọn trong khoảng [0;256]

constexpr int kMaxAudioSampleSize = kFeatureSliceDurationMs*(kAudioSampleFrequency/1000);
// Variables for the model's output categories.
constexpr int kSilenceIndex = 0;
constexpr int kUnknownIndex = 1;
// If you modify the output categories, you need to update the following values.
constexpr int kCategoryCount = 4;
extern const char* kCategoryLabels[kCategoryCount];

#endif  // TENSORFLOW_LITE_MICRO_EXAMPLES_MICRO_SPEECH_MICRO_FEATURES_MICRO_MODEL_SETTINGS_H_
