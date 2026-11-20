// ToF (гэрлийн нисэлтийн хугацааны мэдрэгч)-ийн нэг багцыг нэг мөрөөр дамжуулна.
// 3.3 В тэжээл, 3.3 В логик түвшинтэй бэлэн модуль хэрэглэж газардуулгыг нийлүүлнэ.
// I2C (хоёр утаст цуваа шугам)-ийн SDA нь өгөгдөл, SCL нь цагийн дохио.
// Хоёр модулийн хаяг ижил тул сонгосон нэг модулийг холбоно.
// ESP32-C3 дээр шууд USB хэрэглэхдээ «USB CDC On Boot» сонголтыг идэвхжүүлнэ.
#ifndef USE_VL53L5CX
#define USE_VL53L5CX 0
#endif

#include <Wire.h>
#if USE_VL53L5CX
#include <SparkFun_VL53L5CX_Library.h>
SparkFun_VL53L5CX ranger;
VL53L5CX_ResultsData result;
constexpr char MODEL[] = "VL53L5CX";
constexpr int ZONES = 64;
#else
#include <VL53L1X.h>
VL53L1X ranger;
constexpr char MODEL[] = "VL53L1X";
constexpr int ZONES = 1;
#endif

#if CONFIG_IDF_TARGET_ESP32C3
constexpr int SDA_PIN = 4, SCL_PIN = 5;
#else
constexpr int SDA_PIN = 21, SCL_PIN = 22;
#endif

uint32_t sequenceNumber = 0;
int distances[ZONES];
uint8_t statuses[ZONES];

bool beginRanging() {
#if USE_VL53L5CX
  return ranger.begin() && ranger.setResolution(64) &&
         ranger.setRangingFrequency(10) && ranger.startRanging();
#else
  ranger.setTimeout(150);
  if (!ranger.init() || !ranger.setDistanceMode(VL53L1X::Long) ||
      !ranger.setMeasurementTimingBudget(50000)) return false;
  ranger.startContinuous(60);
  return true;
#endif
}

void setup() {
  Serial.begin(921600);
  Wire.begin(SDA_PIN, SCL_PIN, 400000);
  if (!beginRanging()) {
    Serial.println("Мэдрэгч ассангүй. Сонголт, холболтоо шалгаад хавтанг дахин асаана уу.");
    while (true) delay(1000);
  }
}

void loop() {
#if USE_VL53L5CX
  if (!ranger.isDataReady() || !ranger.getRangingData(&result)) {
    delay(2);
    return;
  }
  for (int i = 0; i < ZONES; ++i) {
    // Линзийн эргэлтийг тооцвол тэг бүс харагдацын баруун дээд буланд тусдаг.
    int zone = (i / 8) * 8 + 7 - (i % 8);
    int target = zone * VL53L5CX_NB_TARGET_PER_ZONE;
    distances[i] = result.distance_mm[target];
    uint8_t raw = result.target_status[target];
    if (!result.nb_target_detected[zone] || raw == 0) statuses[i] = 255;
    else statuses[i] = (raw == 5) ? 0 : raw;
  }
#else
  if (!ranger.dataReady()) {
    delay(2);
    return;
  }
  distances[0] = ranger.read(false);
  statuses[0] = (ranger.timeoutOccurred() || ranger.last_status) ? 255 : ranger.ranging_data.range_status;
#endif
  unsigned long measuredAt = millis();
  Serial.printf("{\"v\":1,\"sensor\":\"%s\",\"seq\":%lu,\"t_ms\":%lu,\"mm\":[",
                MODEL, (unsigned long)sequenceNumber++, measuredAt);
  for (int i = 0; i < ZONES; ++i) {
    if (i) Serial.write(',');
    Serial.print(distances[i]);
  }
  Serial.print("],\"status\":[");
  for (int i = 0; i < ZONES; ++i) {
    if (i) Serial.write(',');
    Serial.print(statuses[i]);
  }
  Serial.println("]}");
}
