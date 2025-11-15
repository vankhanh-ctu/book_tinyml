
#include <SPI.h>
#include <Ethernet.h>
#include "MgsModbus.h"
//#define     ETH_RST        27
#define     ETH_CS         5
#define     ETH_SCLK       18
#define     ETH_MISO       19
#define     ETH_MOSI       23

MgsModbus Mb;
int inByte = 0; // incoming serial byte
const int sensorPin = 35;


// Ethernet settings (depending on MAC and Local network)
byte mac[] = {0x90, 0xA2, 0xDA, 0x0E, 0x94, 0xB5 };
IPAddress ip(192, 168, 0, 120);
IPAddress gateway(192, 168, 0, 1);
//IPAddress gateway(192, 168,0, 10);
IPAddress subnet(255, 255, 255, 0);

void setup() {
  //------serial setup-------------
  pinMode(2, OUTPUT);
  Serial.begin(115200);
  SPI.begin(ETH_SCLK, ETH_MISO, ETH_MOSI);
  Ethernet.init(ETH_CS);
  Ethernet.begin(mac, ip, gateway, subnet);   // start etehrnet interface

  //-------------------Fill MbData------------
  //  Mb.SetBit(0,false);
  Mb.MbData[0] = 0; //choose write from esp--register 40000--
  Mb.MbData[1] = 0; //choose read from esp---register 40001--
  Mb.MbData[2] = 0; //-----------------------register 40002--
  Mb.MbData[3] = 0;
  Mb.MbData[4] = 0;
  Mb.MbData[5] = 0;
  Mb.MbData[6] = 0;
  Mb.MbData[7] = 0;
  Mb.MbData[8] = 0;
  Mb.MbData[9] = 0;
  Mb.MbData[10] = 0;
  Mb.MbData[11] = 0;//--------------------------------40012--
}
void loop() {
  int sensorValue = analogRead(sensorPin);
  Mb.MbData[0] = sensorValue;
  digitalWrite(2, Mb.MbData[1]);
  Mb.MbData[35] = 100;
  //--------------------------------------

  Mb.MbsRun();
}
