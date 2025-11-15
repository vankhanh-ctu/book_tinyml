#ifndef DIFF_H
#define DIFF_H

#include <Arduino.h>

class Differentiator {
public:
    Differentiator(float frequency);
    float computeDerivative(float newValue);

private:
    float signalBuffer[5];
    int signalIndex;
    float frequency;
};

#endif
