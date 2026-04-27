#include <accel_gyro.h>

const int lightSensorPin = IN1;
const int flameSensorPin = IN2;
const int buzzerPin = IN3;

void setup() {
  pinMode(lightSensorPin, INPUT);
  pinMode(flameSensorPin, INPUT);
  pinMode(buzzerPin, OUTPUT);
  
  Serial.begin(9600);  // Инициализация Serial порта на скорости 9600
  
  setupAccel();
  
  Serial.println("System Ready");
  delay(2000);
}

void loop() {
  int lightValue = analogRead(lightSensorPin);
  int flameValue = analogRead(flameSensorPin);
  
  float angleX = readAccelAngle('x', FILTERED);
  float angleY = readAccelAngle('y', FILTERED);
  float angleZ = readAccelAngle('z', FILTERED);
  
  String lightStatus;
  if (lightValue < 300) {
    lightStatus = "DARK";
  } else if (lightValue < 700) {
    lightStatus = "NORM";
  } else {
    lightStatus = "BRIGHT";
  }
  
  String flameStatus;
  if (flameValue > 250) {
    flameStatus = "FIRE!";
    tone(buzzerPin, 1000);
    delay(200);
    noTone(buzzerPin);
    delay(200);
  } else {
    flameStatus = "SAFE";
    noTone(buzzerPin);
  }
  
  // Вывод в Serial порт
  Serial.print("L:");
  Serial.print(lightValue);
  Serial.print(" ");
  Serial.print(lightStatus);
  Serial.print(" | F:");
  Serial.print(flameValue);
  Serial.print(" ");
  Serial.print(flameStatus);
  Serial.print(" | X:");
  Serial.print(int(angleX));
  Serial.print(" Y:");
  Serial.print(int(angleY));
  Serial.print(" Z:");
  Serial.println(int(angleZ));
  
  delay(100);
}
