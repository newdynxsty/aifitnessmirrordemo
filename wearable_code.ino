#include <bluefruit.h>
#include <Wire.h>
#include "SparkFun_BMI270_Arduino_Library.h"
#include "MAX30105.h"
#include "heartRate.h"

// BLE Clients
BLEClientBas clientBas;
BLEClientDis clientDis;
BLEClientUart clientUart;

// Sensor Objects
BMI270 imu;
MAX30105 particleSensor;

// Heart Rate Variables
const byte RATE_SIZE = 5;  // Increase for more stability, decrease for faster response
byte rates[RATE_SIZE];
byte rateSpot = 0;
long lastBeat = 0;  // Time at which the last beat occurred
float beatsPerMinute;
int beatAvg;

// Timing for messages per second
unsigned long lastSendTime = 0;
const unsigned long sendInterval = 50;  // 50ms = 20Hz

void setup() {
  Serial.begin(115200);
  Serial.println("XIAO nRF52840 Sensor to BLE UART Central");

  // 1. Initialize I2C and Sensors
  Wire.begin();

  if (imu.beginI2C() != BMI2_OK) {
    Serial.println("BMI270 not found!");
  }

  if (!particleSensor.begin(Wire, I2C_SPEED_FAST)) {
    Serial.println("MAX30102 not found!");
  }
  particleSensor.setup();  // Configure sensor with default settings

  // 2. Initialize Bluefruit
  Bluefruit.begin(0, 1);
  Bluefruit.setName("XIAO Central");
  clientBas.begin();
  clientDis.begin();
  clientUart.begin();
  clientUart.setRxCallback(bleuart_rx_callback);

  Bluefruit.setConnLedInterval(250);
  Bluefruit.Central.setConnectCallback(connect_callback);
  Bluefruit.Central.setDisconnectCallback(disconnect_callback);

  Bluefruit.Scanner.setRxCallback(scan_callback);
  Bluefruit.Scanner.restartOnDisconnect(true);
  Bluefruit.Scanner.setInterval(160, 80);
  Bluefruit.Scanner.useActiveScan(false);
  Bluefruit.Scanner.start(0);
}

void loop() {
  // 1. CONSTANTLY poll the sensor for pulses (needs high frequency)
  long irValue = particleSensor.getIR();

  if (checkForBeat(irValue) == true) {
    // We sensed a beat!
    long delta = millis() - lastBeat;
    lastBeat = millis();

    beatsPerMinute = 60 / (delta / 1000.0);

    if (beatsPerMinute < 255 && beatsPerMinute > 20) {
      rates[rateSpot++] = (byte)beatsPerMinute;  // Store this reading in the array
      rateSpot %= RATE_SIZE;                     // Wrap variable

      // Take average of readings
      beatAvg = 0;
      for (byte x = 0; x < RATE_SIZE; x++) beatAvg += rates[x];
      beatAvg /= RATE_SIZE;
    }
  }

  // 2. PERIODICALLY send data
  if (Bluefruit.Central.connected() && clientUart.discovered()) {
    unsigned long currentMillis = millis();

    // Limit to 2 messages per second
    if (currentMillis - lastSendTime >= sendInterval) {
      lastSendTime = currentMillis;

      // Read BMI270 IMU
      imu.getSensorData();
      float ax = imu.data.accelX;
      float ay = imu.data.accelY;
      float az = imu.data.accelZ;

      // Construct Message String
      // Format: A:X.X,Y.Y,Z.Z|HR:Value
      char payload[64];
      // If IR is too low, user probably isn't touching the sensor
      if (irValue < 50000) {
        snprintf(payload, sizeof(payload), "A:%.2f,%.2f,%.2f|HR:No\n", ax, ay, az);
      } else {
        snprintf(payload, sizeof(payload), "A:%.2f,%.2f,%.2f|HR:%d\n", ax, ay, az, beatAvg);
      }

      // Send via BLE UART
      clientUart.print(payload);

      // Local Debug
      // Serial.print("Sending: ");
      // Serial.print(payload);
    }
  }
}

// --- BLE CALLBACKS (Keep your original logic below) ---

void scan_callback(ble_gap_evt_adv_report_t* report) {
  if (Bluefruit.Scanner.checkReportForService(report, clientUart)) {
    Bluefruit.Central.connect(report);
  } else {
    Bluefruit.Scanner.resume();
  }
}

void connect_callback(uint16_t conn_handle) {
  Serial.println("Connected");
  clientDis.discover(conn_handle);
  clientBas.discover(conn_handle);
  if (clientUart.discover(conn_handle)) {
    clientUart.enableTXD();
  } else {
    Bluefruit.disconnect(conn_handle);
  }
}

void disconnect_callback(uint16_t conn_handle, uint8_t reason) {
  Serial.print("Disconnected, reason = 0x");
  Serial.println(reason, HEX);
}

void bleuart_rx_callback(BLEClientUart& uart_svc) {
  while (uart_svc.available()) {
    Serial.print((char)uart_svc.read());
  }
}