#include "VL53L1X.h"

VL53L1X sensor;

void sensor_init(VL53L1X::DistanceMode range_mode, bool high_speed) {
  Wire.begin();
  sensor.setTimeout(500);
  sensor.init();
  sensor.setDistanceMode(range_mode);  
  int budget = high_speed ? 33000 : 140000;
  sensor.setMeasurementTimingBudget(budget);
}

void setup() {
  Serial.begin(9600);
  // range_mode: VL53L1X::Short, VL53L1X::Medium, or VL53L1X::Long
  sensor_init(VL53L1X::Medium, false);   
}

void loop() {
  int dist = sensor.readRangeSingleMillimeters();
  Serial.println(dist);
  delay(1000);
}