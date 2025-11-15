#include <Arduino.h>
#include "Filter.h"
float y_n[] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
float x_n[] = {0, 0, 0, 0, 0, 0, 0, 0};
double omega0;
double dt;
double Q;
double domega;
double tn1 = 0;
double alpha[] = {1, 2.6231, 3.4142, 2.6231, 1};
double a_low[] = {3.18725516, -3.87603454 , 2.12354265, -0.44123371};
double b_low[] = {0.0004044,  0.00161761, 0.00242641, 0.00161761 , 0.0004044};
double a_high[] = {0.04579258, -0.48669163,  0.00960712, -0.01770531};
double b_high[] = { 0.09748729, -0.38994916 , 0.58492374, -0.38994916 , 0.09748729};
double a_band[] = {    -3.16578702, -4.41793298, -3.02814711, -0.91497583};
double b_band[] = { 0.00094469,  0,         -0.00188938,  0,          0.00094469};
double a_stop[] = {     3.16578702, -4.41793298, 3.02814711, -0.91497583 };
double b_stop[] = {0.95654323, -3.09696707, 4.41982236, -3.09696707, 0.95654323 };
//for moving average*********************************************
void highpass_init(float f0, float fs)
{
  omega0 = 6.28318530718 * f0;
  dt = 1.0 / fs;
  tn1 = -dt;
  for (int k = 0; k < 4 + 1; k++) {
    x_n[k] = 0;
    y_n[k] = 0;
  }
  //float alpha = omega0*dt;
  float beta = omega0 * dt;
  float betasq = beta * beta;
  // float c[] = {omega0*omega0, sqrt(2)*omega0, 1};
  float D = alpha[0] * betasq * betasq + 2 * alpha[1] * beta * betasq + 4 * alpha[2] * betasq + 8 * alpha[3] * beta + 16 * alpha[4];
  b_high[0] = 16.0 / D;
  b_high[1] = -64.0 / D;
  b_high[2] = 96.0 / D;
  b_high[3] = -64.0 / D;
  b_high[4] = 16.0 / D;
  a_high[0] = -(4 * alpha[0] * betasq * betasq + 4 * alpha[1] * beta * betasq - 16 * alpha[3] * beta - 64 * alpha[4]) / D;
  a_high[1] = -(6 * alpha[0] * betasq * betasq - 8 * alpha[2] * betasq + 96 * alpha[4]) / D;
  a_high[2] = -(4 * alpha[0] * betasq * betasq - 4 * alpha[1] * beta * betasq + 16 * alpha[3] * beta - 64 * alpha[4]) / D;
  a_high[3] = -(alpha[0] * betasq * betasq - 2 * alpha[1] * beta * betasq + 4 * alpha[2] * betasq - 8 * alpha[3] * beta + 16 * alpha[4]) / D;

}
void lowpass_init(float f0, float fs)
{
  omega0 = 6.28318530718 * f0;
  dt = 1.0 / fs;
  tn1 = -dt;
  for (int k = 0; k < 4 + 1; k++) {
    x_n[k] = 0;
    y_n[k] = 0;
  }
  //float alpha = omega0*dt;
  float beta = omega0 * dt;
  float betasq = beta * beta;
  // float c[] = {omega0*omega0, sqrt(2)*omega0, 1};
  float D = alpha[0] * betasq * betasq + 2 * alpha[1] * beta * betasq + 4 * alpha[2] * betasq + 8 * alpha[3] * beta + 16 * alpha[4];
  b_low[0] = (betasq * betasq) / D;
  b_low[1] = (4 * betasq * betasq) / D;
  b_low[2] = (6 * betasq * betasq) / D;
  b_low[3] = (4 * betasq * betasq) / D;
  b_low[4] = (betasq * betasq) / D;
  a_low[0] = -(4 * alpha[0] * betasq * betasq + 4 * alpha[1] * beta * betasq - 16 * alpha[3] * beta - 64 * alpha[4]) / D;
  a_low[1] = -(6 * alpha[0] * betasq * betasq - 8 * alpha[2] * betasq + 96 * alpha[4]) / D;
  a_low[2] = -(4 * alpha[0] * betasq * betasq - 4 * alpha[1] * beta * betasq + 16 * alpha[3] * beta - 64 * alpha[4]) / D;
  a_low[3] = -(alpha[0] * betasq * betasq - 2 * alpha[1] * beta * betasq + 4 * alpha[2] * betasq - 8 * alpha[3] * beta + 16 * alpha[4]) / D;
}

void bandpass_init(float f0, float fw, float fs)
{
  // f0: central frequency (Hz)
  // fw: bandpass width (Hz)
  // fs: sample frequency (Hz)
  double fh = f0 + fw / 2;
  double fl = f0 - fw / 2;

  omega0 = 6.28318530718 * f0;
  domega = 6.28318530718 * (fw);
  dt = 1.0 / fs;
  Q = omega0 / domega;
  tn1 = -dt;
  for (int k = 0; k < 4 + 1; k++) {
    x_n[k] = 0;
    y_n[k] = 0;
  }

  double beta = omega0 * dt;
  double betasq = beta * beta;
  double D = 16 + (8 * sqrt(2) * beta) / Q + 4 * betasq * (2 + 1 / (Q * Q)) + (2 * sqrt(2) * beta * betasq) / Q + betasq * betasq;

  b_band[0] = ((4 / (Q * Q)) * betasq) / D;
  b_band[1] = 0;
  b_band[2] = ((-8 / (Q * Q)) * betasq) / D;;
  b_band[3] = 0;
  b_band[4] = ((4 / (Q * Q)) * betasq) / D;

  a_band[0] = -(-64 - (16 * sqrt(2) * beta) / Q + (4 * sqrt(2) * betasq * beta) / Q + 4 * betasq * betasq) / D;
  a_band[1] = -(96 - 8 * (2 + 1 / (Q * Q)) * betasq + 6 * betasq * betasq) / D ;
  a_band[2] = -(-64 + (16 * sqrt(2) * beta) / Q - (4 * sqrt(2) * betasq * beta) / Q + 4 * betasq * betasq) / D;
  a_band[3] = -(16 - (8 * sqrt(2) * beta) / Q + 4 * betasq * (2 + 1 / (Q * Q)) - (2 * sqrt(2) * beta * betasq) / Q + betasq * betasq) / D;


}

void bandstop_init(float f0, float fw, float fs)
{
  // f0: central frequency (Hz)
  // fw: bandpass width (Hz)
  // fs: sample frequency (Hz)
  double fh = f0 + fw / 2;
  double fl = f0 - fw / 2;

  omega0 = 6.28318530718 * f0;
  domega = 6.28318530718 * (fw + 00);
  dt = 1.0 / fs;
  Q = omega0 / domega;
  tn1 = -dt;
  for (int k = 0; k < 4 + 1; k++) {
    x_n[k] = 0;
    y_n[k] = 0;
  }

  double beta = omega0 * dt;
  double betasq = beta * beta;

  double D = 16 + (8 * sqrt(2) * beta) / Q + 4 * betasq * (2 + 1 / (Q * Q)) + (2 * sqrt(2) * beta * betasq) / Q + betasq * betasq;


  b_stop[0] = (16 + 8 * beta * beta + betasq * betasq) / D;
  b_stop[1] = (-64 + 4 * betasq * betasq) / D;
  b_stop[2] = (96 - 16 * betasq + 6 * betasq * betasq) / D;
  b_stop[3] = (-64 + 4 * betasq * betasq) / D;
  b_stop[4] = (16 + 8 * beta * beta + betasq * betasq) / D;

  a_stop[0] = -(-64 - (16 * sqrt(2) * beta) / Q + (4 * sqrt(2) * betasq * beta) / Q + 4 * betasq * betasq) / D;
  a_stop[1] = -(96 - 8 * (2 + 1 / (Q * Q)) * betasq + 6 * betasq * betasq) / D ;
  a_stop[2] = -(-64 + (16 * sqrt(2) * beta) / Q - (4 * sqrt(2) * betasq * beta) / Q + 4 * betasq * betasq) / D;
  a_stop[3] = -(16 - (8 * sqrt(2) * beta) / Q + 4 * betasq * (2 + 1 / (Q * Q)) - (2 * sqrt(2) * beta * betasq) / Q + betasq * betasq) / D;


}
//***************************************************************
float lowpass(float x1)
{
  y_n[4] = y_n[3];
  y_n[3] = y_n[2];
  y_n[2] = y_n[1];
  y_n[1] = y_n[0];
  y_n[0] = 0;
  y_n[0] = a_low[0] * y_n[1] +  a_low[1] * y_n[2] +  a_low[2] * y_n[3] +  a_low[3] * y_n[4] + b_low[0] * x1 + b_low[1] * x_n[0] + b_low[2] * x_n[1] + b_low[3] * x_n[2] + b_low[4] * x_n[3];
  x_n[3] = x_n[2];
  x_n[2] = x_n[1];
  x_n[1] = x_n[0];
  x_n[0] = x1;
  return y_n[0];

}
float highpass(float x1)
{
  y_n[4] = y_n[3];
  y_n[3] = y_n[2];
  y_n[2] = y_n[1];
  y_n[1] = y_n[0];
  y_n[0] = a_high[0] * y_n[1] +  a_high[1] * y_n[2] +  a_high[2] * y_n[3] +  a_high[3] * y_n[4] + b_high[0] * x1 + b_high[1] * x_n[0] + b_high[2] * x_n[1] + b_high[3] * x_n[2] + b_high[4] * x_n[3];
  x_n[3] = x_n[2];
  x_n[2] = x_n[1];
  x_n[1] = x_n[0];
  x_n[0] = x1;
  return y_n[0];
}

float bandpass(float x1)
{

  y_n[4] = y_n[3];
  y_n[3] = y_n[2];
  y_n[2] = y_n[1];
  y_n[1] = y_n[0];
  y_n[0] =    a_band[0] * y_n[1] +  a_band[1] * y_n[2] + a_band[2] * y_n[3] + a_band[3] * y_n[4] +  b_band[0] * x1 + b_band[1] * x_n[0] + b_band[2] * x_n[1]
              + b_band[3] * x_n[2] + b_band[4] * x_n[3] ;

  x_n[3] = x_n[2];
  x_n[2] = x_n[1];
  x_n[1] = x_n[0];
  x_n[0] = x1;
  return y_n[0];
}

float bandstop(float x1)
{
  //y[0] = a[0]*y[0] +a[1]*y[1]+ a[2]*y[2] + b[0]*yes_features_data[i] + b[1]*store[0] + b[2]*store[1];

  y_n[4] = y_n[3];
  y_n[3] = y_n[2];
  y_n[2] = y_n[1];
  y_n[1] = y_n[0];
  y_n[0] =    a_stop[0] * y_n[1] +  a_stop[1] * y_n[2] + a_stop[2] * y_n[3] + a_stop[3] * y_n[4] +  b_stop[0] * x1 + b_stop[1] * x_n[0] + b_stop[2] * x_n[1]
              + b_stop[3] * x_n[2] + b_stop[4] * x_n[3] ;

  x_n[3] = x_n[2];
  x_n[2] = x_n[1];
  x_n[1] = x_n[0];
  x_n[0] = x1;
  return y_n[0];
}
float movingAvg(float *ptrArrNumbers, double *ptrSum, int pos, int len, float nextNum)
{
  //Subtract the oldest number from the prev sum, add the new number
  *ptrSum = *ptrSum - ptrArrNumbers[pos] + nextNum;
  //Assign the nextNum to the position in the array
  ptrArrNumbers[pos] = nextNum;
  //return the average
  return *ptrSum / len;
}

void reset_filter()
{
  y_n[0] = y_n[1] = y_n[2] = y_n[3] = y_n[4] = y_n[5] = y_n[6] = y_n[7] = y_n[8] = 0;
  x_n[0] = x_n[1] = x_n[2] = x_n[3] = x_n[4] = x_n[5] = x_n[6] = x_n[7] = 0;
}
