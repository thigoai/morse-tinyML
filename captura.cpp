#include <Arduino.h>

#define BUTTON_PIN  13  
#define BUZZER_PIN  27  

#define DEBOUNCE_MS        50
#define NOISE_MS           50
#define DOT_LIMIT_MS       400
#define END_OF_LETTER_PAUSE_MS 500

unsigned long pressStartTime = 0;
unsigned long releaseTime = 0;
bool isPressed = false;

void setup() {
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  pinMode(BUZZER_PIN, OUTPUT);
  digitalWrite(BUZZER_PIN, LOW);

  Serial.begin(115200);
  Serial.println("Tempo_ms,Classe");
}

void loop() {
  bool currentState = digitalRead(BUTTON_PIN) == LOW;

  if (currentState && !isPressed) {
    delay(DEBOUNCE_MS);
    if (digitalRead(BUTTON_PIN) == LOW) {
      pressStartTime = millis();
      isPressed = true;
    }
  }

  if (!currentState && isPressed) {
    delay(DEBOUNCE_MS);
    if (digitalRead(BUTTON_PIN) == HIGH) {
      unsigned long pressDuration = millis() - pressStartTime;

      Serial.print("PRESS_DURATION_MS: ");
      Serial.println(pressDuration);

      releaseTime = millis();
      isPressed = false;
    }
  }

  if (!isPressed && releaseTime > 0) {
    unsigned long pauseDuration = millis() - releaseTime;
    if (pauseDuration > END_OF_LETTER_PAUSE_MS) {

      Serial.println("FIM DE LETRA");
      releaseTime = 0;
    }
  }
}

void beep(int durationMs) {
  digitalWrite(BUZZER_PIN, HIGH);
  delay(durationMs);
  digitalWrite(BUZZER_PIN, LOW);
}