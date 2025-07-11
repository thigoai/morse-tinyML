#include <Arduino.h>

#define BUTTON_PIN  13  
#define BUZZER_PIN  22    

#define DEBOUNCE_MS        50   // Debounce time for button press/release
#define NOISE_MS           50  // Duration below which a press is considered noise
#define DOT_LIMIT_MS       400  // Duration below which a press is a "dot"
#define END_OF_LETTER_PAUSE_MS 1500 // Duration of pause to signify end of a letter

unsigned long pressStartTime = 0; // Stores the time when the button was pressed
unsigned long releaseTime = 0;    // Stores the time when the button was released
bool isPressed = false;           // Flag to track if the button is currently pressed

void setup() {
  pinMode(BUTTON_PIN, INPUT_PULLUP); // Set button pin as input with internal pull-up resistor
  pinMode(BUZZER_PIN, OUTPUT);       // Set buzzer pin as output
  digitalWrite(BUZZER_PIN, LOW);     // Ensure buzzer is off initially

  Serial.begin(115200);              // Initialize serial communication
  Serial.println("Tempo_ms,Classe"); // Print header for CSV output
}

void loop() {
  bool currentState = digitalRead(BUTTON_PIN) == LOW; // Read the current state of the button (LOW when pressed)

  // --- Button Press Detection ---
  if (currentState && !isPressed) {
    // Button is now pressed, and it wasn't pressed before
    delay(DEBOUNCE_MS); // Wait for debounce
    if (digitalRead(BUTTON_PIN) == LOW) { // Check again after debounce
      pressStartTime = millis(); // Record the press start time
      isPressed = true;          // Set the pressed flag
    }
  }

  // --- Button Release Detection and Classification ---
  if (!currentState && isPressed) {
    // Button is now released, and it was pressed before
    delay(DEBOUNCE_MS); // Wait for debounce
    if (digitalRead(BUTTON_PIN) == HIGH) { // Check again after debounce
      unsigned long pressDuration = millis() - pressStartTime; // Calculate press duration
    

      // Print for export as dataset
      Serial.print(pressDuration);
      Serial.print(" ");

      releaseTime = millis(); // Record the release time
      isPressed = false;      // Reset the pressed flag
    }
  }

  // --- Long Pause Detection (End of Letter) ---
  // This check should only happen if the button is *not* currently pressed,
  // and a release time has been recorded (meaning a previous press occurred).
  if (!isPressed && releaseTime > 0) {
    unsigned long pauseDuration = millis() - releaseTime; // Calculate pause duration
    if (pauseDuration > END_OF_LETTER_PAUSE_MS) {
      Serial.println(""); // Indicate end of letter
      releaseTime = 0; // Reset releaseTime to prevent repeated "END OF LETTER" messages
                       // for the same long pause
    }
  }
}

// Beep function for short/long feedback
void beep(int durationMs) {
  digitalWrite(BUZZER_PIN, HIGH); // Turn buzzer on
  delay(durationMs);              // Keep it on for the specified duration
  digitalWrite(BUZZER_PIN, LOW);  // Turn buzzer off
}