import socket
import sys

ESP32_IP = "192.168.4.1"   # default AP IP of ESP32 in AP mode
ESP32_PORT = 4210
TIMEOUT_SEC = 2.0

def send_command(command: str):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(TIMEOUT_SEC)

    try:
        sock.sendto(command.encode("utf-8"), (ESP32_IP, ESP32_PORT))
        print(f"Sent: {command}")

        data, addr = sock.recvfrom(1024)
        print(f"Reply from {addr}: {data.decode('utf-8', errors='ignore')}")
    except socket.timeout:
        print("No reply received (timeout).")
    finally:
        sock.close()

def main():
    if len(sys.argv) < 2:
        print("Usage examples:")
        print("  python valve_client.py OPEN")
        print("  python valve_client.py CLOSE")
        print("  python valve_client.py STATUS")
        print("  python valve_client.py STOP")
        print("  python valve_client.py OPEN_MS 1500")
        return

    if sys.argv[1].upper() == "OPEN_MS":
        if len(sys.argv) != 3:
            print("Usage: python valve_client.py OPEN_MS <duration_ms>")
            return
        command = f"OPEN_MS {sys.argv[2]}"
    else:
        command = sys.argv[1].upper()

    send_command(command)

if __name__ == "__main__":
    main()