#include "diff.h"

Differentiator::Differentiator(float freq) : frequency(freq), signalIndex(0) {
    for (int i = 0; i < 5; i++) {
        signalBuffer[i] = 0;
    }
}

float Differentiator::computeDerivative(float newValue) {
    for (int i = 4; i > 0; i--) {
        signalBuffer[i] = signalBuffer[i - 1];
    }
    signalBuffer[0] = newValue;
    float derivative = 0;
    if (signalIndex >= 4) {
        derivative = (signalBuffer[0] - signalBuffer[4]) * frequency / 8;
    }

    signalIndex++;
    return derivative;
}
