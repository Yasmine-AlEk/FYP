#include <WiFi.h>
#include <WiFiUdp.h>

// ================= Wi-Fi settings =================
const char* WIFI_SSID = "yy";
const char* WIFI_PASS = "00000011";
const bool USE_AP_MODE = false;

const uint16_t UDP_PORT = 4210;

// ================= Hardware settings =================
const int PUMP_PIN  = 25;
const int VALVE_PIN = 26;

// true  -> relay turns ON when pin is HIGH
// false -> relay turns ON when pin is LOW
const bool PUMP_ACTIVE_HIGH  = true;
const bool VALVE_ACTIVE_HIGH = true;

// pump on -> wait 2 s -> valve on
const uint32_t START_DELAY_MS = 2000;

// watchdog timeout
const uint32_t WATCHDOG_MS = 5000;

// ================= Globals =================
WiFiUDP udp;
char rxBuf[256];

bool pumpOn = false;
bool valveOn = false;

uint32_t lastCmdMs = 0;

enum SequenceState {
  SEQ_NONE,
  SEQ_START_WAIT_VALVE_ON
};

SequenceState seqState = SEQ_NONE;
uint32_t seqDeadlineMs = 0;

// ---------------- Helpers ----------------
void writePumpRaw(bool on) {
  if (PUMP_ACTIVE_HIGH) digitalWrite(PUMP_PIN, on ? HIGH : LOW);
  else                  digitalWrite(PUMP_PIN, on ? LOW  : HIGH);
}

void writeValveRaw(bool on) {
  if (VALVE_ACTIVE_HIGH) digitalWrite(VALVE_PIN, on ? HIGH : LOW);
  else                   digitalWrite(VALVE_PIN, on ? LOW  : HIGH);
}

void setPump(bool on) {
  pumpOn = on;
  writePumpRaw(on);
}

void setValve(bool on) {
  valveOn = on;
  writeValveRaw(on);
}

void cancelSequences() {
  seqState = SEQ_NONE;
}

void allOff() {
  cancelSequences();
  setValve(false);
  setPump(false);
}

void touchWatchdog() {
  lastCmdMs = millis();
}

void replyToSender(IPAddress ip, uint16_t port, const String& msg) {
  udp.beginPacket(ip, port);
  udp.print(msg);
  udp.endPacket();
}

String getDeviceIP() {
  if (USE_AP_MODE) return WiFi.softAPIP().toString();
  return WiFi.localIP().toString();
}

String normalize(String s) {
  s.trim();
  s.toUpperCase();
  return s;
}

String statusLine() {
  String s = "IP=" + getDeviceIP();
  s += " PUMP=" + String(pumpOn ? "ON" : "OFF");
  s += " VALVE=" + String(valveOn ? "ON" : "OFF");
  s += " SEQ=" + String(seqState == SEQ_START_WAIT_VALVE_ON ? "START_WAIT" : "NONE");
  s += " WD_MS=" + String(WATCHDOG_MS);
  return s;
}

// ---------------- Setup ----------------
void setup() {
  Serial.begin(115200);
  delay(500);

  // safe OFF immediately on boot
  pinMode(PUMP_PIN, OUTPUT);
  pinMode(VALVE_PIN, OUTPUT);
  writePumpRaw(false);
  writeValveRaw(false);

  pumpOn = false;
  valveOn = false;
  cancelSequences();
  touchWatchdog();

  if (USE_AP_MODE) {
    WiFi.mode(WIFI_AP);
    bool ok = WiFi.softAP(WIFI_SSID, WIFI_PASS);
    Serial.println(ok ? "AP started" : "AP start FAILED");
    Serial.print("AP IP: ");
    Serial.println(WiFi.softAPIP());
  } else {
    WiFi.mode(WIFI_STA);
    WiFi.begin(WIFI_SSID, WIFI_PASS);

    Serial.print("Connecting to Wi-Fi");
    while (WiFi.status() != WL_CONNECTED) {
      delay(300);
      Serial.print(".");
    }
    Serial.println();
    Serial.println("Connected");
    Serial.print("STA IP: ");
    Serial.println(WiFi.localIP());
  }

  udp.begin(UDP_PORT);
  Serial.print("UDP listening on port ");
  Serial.println(UDP_PORT);

  Serial.println("Commands:");
  Serial.println("  s       -> pump on, wait 2s, valve on");
  Serial.println("  x       -> valve off, pump off");
  Serial.println("  k       -> emergency stop");
  Serial.println("  status  -> status");
  Serial.println("  hb      -> heartbeat");
  Serial.println("  p1 / p0");
  Serial.println("  v1 / v0");
}

// ---------------- Loop ----------------
void loop() {
  uint32_t now = millis();

  // delayed valve-on for start sequence
  if (seqState == SEQ_START_WAIT_VALVE_ON && now >= seqDeadlineMs) {
    setValve(true);
    seqState = SEQ_NONE;
    Serial.println("s complete -> valve on");
  }

  // watchdog safety
  if ((pumpOn || valveOn || seqState != SEQ_NONE) && (now - lastCmdMs > WATCHDOG_MS)) {
    Serial.println("WATCHDOG -> all off");
    allOff();
  }

  int packetSize = udp.parsePacket();
  if (packetSize <= 0) {
    delay(5);
    return;
  }

  int len = udp.read(rxBuf, sizeof(rxBuf) - 1);
  if (len <= 0) return;
  rxBuf[len] = '\0';

  IPAddress senderIP = udp.remoteIP();
  uint16_t senderPort = udp.remotePort();

  String raw = String(rxBuf);
  String cmd = normalize(raw);

  Serial.print("RX from ");
  Serial.print(senderIP);
  Serial.print(":");
  Serial.print(senderPort);
  Serial.print(" -> ");
  Serial.println(raw);

  // high-level commands
  if (cmd == "S") {
    touchWatchdog();
    cancelSequences();
    setPump(true);
    seqState = SEQ_START_WAIT_VALVE_ON;
    seqDeadlineMs = millis() + START_DELAY_MS;
    replyToSender(senderIP, senderPort, "ACK_S");
    return;
  }

  if (cmd == "X") {
    touchWatchdog();
    cancelSequences();
    setValve(false);
    setPump(false);
    replyToSender(senderIP, senderPort, "ACK_X");
    return;
  }

  if (cmd == "K") {
    touchWatchdog();
    allOff();
    replyToSender(senderIP, senderPort, "ACK_K");
    return;
  }

  if (cmd == "STATUS") {
    touchWatchdog();
    replyToSender(senderIP, senderPort, statusLine());
    return;
  }

  if (cmd == "HB") {
    touchWatchdog();
    return;   // no reply needed
  }

  // manual commands
  if (cmd == "P1") {
    touchWatchdog();
    cancelSequences();
    setPump(true);
    replyToSender(senderIP, senderPort, "ACK_P1");
    return;
  }

  if (cmd == "P0") {
    touchWatchdog();
    cancelSequences();
    setPump(false);
    replyToSender(senderIP, senderPort, "ACK_P0");
    return;
  }

  if (cmd == "V1") {
    touchWatchdog();
    cancelSequences();
    setValve(true);
    replyToSender(senderIP, senderPort, "ACK_V1");
    return;
  }

  if (cmd == "V0") {
    touchWatchdog();
    cancelSequences();
    setValve(false);
    replyToSender(senderIP, senderPort, "ACK_V0");
    return;
  }

  replyToSender(senderIP, senderPort, "ERR_UNKNOWN_CMD");
}