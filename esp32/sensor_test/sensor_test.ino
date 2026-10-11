#include "DHT.h"

#define SOIL1_PIN 32
#define SOIL2_PIN 33
#define WATER_PIN 34
#define RAIN_PIN 27

#define DHT_PIN 4
#define DHT_TYPE DHT22

DHT dht(DHT_PIN, DHT_TYPE);

void setup() {
  Serial.begin(115200);
  pinMode(RAIN_PIN, INPUT);
  dht.begin();
  delay(2000);
}

void loop() {
  int soil1 = analogRead(SOIL1_PIN);
  int soil2 = analogRead(SOIL2_PIN);
  int water = analogRead(WATER_PIN);
  int rain = digitalRead(RAIN_PIN);

  float temperature = dht.readTemperature();
  float humidity = dht.readHumidity();

  Serial.print("{\"soil1_raw\":"); Serial.print(soil1);
  Serial.print(",\"soil2_raw\":"); Serial.print(soil2);
  Serial.print(",\"water_level_raw\":"); Serial.print(water);
  Serial.print(",\"rain_pin_raw\":"); Serial.print(rain);
  Serial.print(",\"rain_detected\":");
  Serial.print(rain == LOW ? "true" : "false");

  Serial.print(",\"temperature_c\":");
  if (isnan(temperature)) Serial.print("null");
  else Serial.print(temperature, 1);

  Serial.print(",\"humidity_pct\":");
  if (isnan(humidity)) Serial.print("null");
  else Serial.print(humidity, 1);

  Serial.println("}");
  delay(2500);
}
