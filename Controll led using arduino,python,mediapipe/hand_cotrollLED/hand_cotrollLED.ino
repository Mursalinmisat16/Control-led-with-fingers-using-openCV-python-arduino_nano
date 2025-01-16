int ledPins[] = {3, 4, 5, 6, 7};  // Thumb, Index, Middle, Ring, Pinky

void setup() {
  Serial.begin(9600);
  for (int i = 0; i < 5; i++) {
    pinMode(ledPins[i], OUTPUT);
  }
}

void loop() {
  if (Serial.available() >= 5) {  // Expect 5 bytes for 5 fingers
    for (int i = 0; i < 5; i++) {
      int fingerState = Serial.read();  // Read each finger state (0 or 1)
      digitalWrite(ledPins[i], fingerState == 1 ? HIGH : LOW);  // Turn LED on/off
    }
  }
}
