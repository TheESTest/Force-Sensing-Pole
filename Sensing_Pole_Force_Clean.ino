/*Elation Sports Technologies LLC
8 Oct 2021

Sensing Pole - Strain Gauges (Force-Sensing)
https://github.com/TheESTest/Force-Sensing-Pole

This script reads 2 x half Wheatstone bridge circuits,
each connected to a Sparkfun HX711 breakout board.

For more project ideas, check us out online!
www.elationsportstechnologies.com/blog

This code builds on the libraries created by Olav Kallhovd:
https://github.com/olkal/HX711_ADC

MIT License

Copyright (c) 2017 Olav Kallhovd

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

*/

#include <Arduino.h>
#include <HX711_ADC.h>

unsigned long t = 0;
String str_to_send;

//Pins:
const int HX711_dout_1 = 2; //mcu > HX711 dout pin
const int HX711_sck_1 = 3; //mcu > HX711 sck pin
const int HX711_dout_2 = 4; //mcu > HX711 dout pin
const int HX711_sck_2 = 5; //mcu > HX711 sck pin

//HX711 constructor:
HX711_ADC LoadCell_1(HX711_dout_1, HX711_sck_1);
HX711_ADC LoadCell_2(HX711_dout_2, HX711_sck_2);

void setup(){

  Serial.begin(9600);

  //Default calibration values that were initially used for scaling
  //the data. These are otherwise unused, so set them to whatever value you wish.
  float calibrationValue_1;
  float calibrationValue_2;
  calibrationValue_1 = 50500;
  calibrationValue_2 = 50500;

  unsigned long stabilizingtime = 3000;
  delay(1000);

  LoadCell_1.begin();
  LoadCell_2.begin();
  byte loadcell_1_rdy = 0;
  byte loadcell_2_rdy = 0;

  Serial.println("Initializing load cells.");
  delay(1000);

  while ((loadcell_1_rdy + loadcell_2_rdy) < 2) {
    if (!loadcell_1_rdy){
      loadcell_1_rdy = LoadCell_1.startMultiple(stabilizingtime, false);
    }
    if (!loadcell_2_rdy){
      loadcell_2_rdy = LoadCell_2.startMultiple(stabilizingtime, false);
    }
  }

  if (LoadCell_1.getTareTimeoutFlag()) {
    Serial.println("Timeout - check MCU-to-HX711 #1 wiring, pin assignments, etc");
    while (1==1){
      0;
    }
  }
  else{
    Serial.println("LoadCell_1 setup is complete.");
    delay(500);
  }
  if (LoadCell_2.getTareTimeoutFlag()) {
    Serial.println("Timeout - check MCU-to-HX711 #2 wiring, pin assignments, etc");
    while (1==1){
      0;
    }
  }
  else{
    Serial.println("LoadCell_2 setup is complete");
    delay(500);
  }
  
  LoadCell_1.setCalFactor(calibrationValue_1);
  LoadCell_2.setCalFactor(calibrationValue_2);
  
  Serial.println("Setup completed successfully.");

}

void loop(){

  const int serialPrintInterval = 0; //increase value to slow down serial print activity

  val_1 = analogRead(analog_pin_1);
  val_2 = analogRead(analog_pin_2);
  //Serial.println(val);

  static boolean newDataReady_1 = 0;
  if (LoadCell_1.update()){
    newDataReady_1 = true;// check for new data/start next conversion
  }
  static boolean newDataReady_2 = 0;
  if (LoadCell_2.update()){
    newDataReady_2 = true;// check for new data/start next conversion
  }

  if (newDataReady_1 && newDataReady_2) {
    if (millis() > t + serialPrintInterval) {
      str_to_send = "0:" + String(LoadCell_1.getData(), 8) + ", 1:" + String(LoadCell_2.getData(), 8) + ", 2:" + String(val_1) + ", 3:" + String(val_2);
      Serial.print(str_to_send);
      Serial.print('\n');
      newDataReady_1 = 0;
      t = millis();
    }
  }

}
