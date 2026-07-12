#include <Servo.h>

Servo gripper;
const int SERVO_PIN = 9;

// Tune these
const float MIN_POS_M = 0.0;       // closed
const float MAX_POS_M = 0.08925;   // open (example)
const int SERVO_MIN_US = 900;
const int SERVO_MAX_US = 2100;

// Optional servo feedback (if available on your hardware).
const bool USE_SERVO_FEEDBACK = false;
const int SERVO_FEEDBACK_PIN = A0;
const int FEEDBACK_MIN_ADC = 200;
const int FEEDBACK_MAX_ADC = 850;

String line;
float current_norm = 1.0;  // starts open in this sketch

int mapPosToUs(float pos_m) {
  if (pos_m < MIN_POS_M) pos_m = MIN_POS_M;
  if (pos_m > MAX_POS_M) pos_m = MAX_POS_M;
  float t = (pos_m - MIN_POS_M) / (MAX_POS_M - MIN_POS_M);  // 0..1
  return (int)(SERVO_MIN_US + t * (SERVO_MAX_US - SERVO_MIN_US));
}

float readFeedbackNormalized() {
  if (!USE_SERVO_FEEDBACK) {
    return current_norm;
  }
  int raw = analogRead(SERVO_FEEDBACK_PIN);
  if (raw < FEEDBACK_MIN_ADC) raw = FEEDBACK_MIN_ADC;
  if (raw > FEEDBACK_MAX_ADC) raw = FEEDBACK_MAX_ADC;
  float t = (float)(raw - FEEDBACK_MIN_ADC) / (float)(FEEDBACK_MAX_ADC - FEEDBACK_MIN_ADC);
  if (t < 0.0) t = 0.0;
  if (t > 1.0) t = 1.0;
  return t;
}

void setup() {
  Serial.begin(115200);
  gripper.attach(SERVO_PIN, SERVO_MIN_US, SERVO_MAX_US);
  gripper.writeMicroseconds(mapPosToUs(MAX_POS_M)); // start open
  current_norm = 1.0;
  if (USE_SERVO_FEEDBACK) {
    pinMode(SERVO_FEEDBACK_PIN, INPUT);
  }
}

void loop() {
  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\n') {
      line.trim();

      // Query current normalized position.
      if (line == "Q" || line == "?") {
        Serial.println(readFeedbackNormalized(), 6);
        line = "";
        continue;
      }

      // Expect: P <float>
      if (line.length() > 2 && line[0] == 'P') {
        float pos = line.substring(1).toFloat();
        float t = (pos - MIN_POS_M) / (MAX_POS_M - MIN_POS_M);
        if (t < 0.0) t = 0.0;
        if (t > 1.0) t = 1.0;
        current_norm = t;
        gripper.writeMicroseconds(mapPosToUs(pos));
        Serial.println("OK"); // optional ack
      } else {
        // Also accept direct normalized command for compatibility.
        float t = line.toFloat();
        if (t < 0.0) t = 0.0;
        if (t > 1.0) t = 1.0;
        float pos = MIN_POS_M + t * (MAX_POS_M - MIN_POS_M);
        gripper.writeMicroseconds(mapPosToUs(pos));
        current_norm = t;
      }
      line = "";
    } else {
      line += c;
    }
  }
}