import atexit
import os
import signal
import socket
import sys
import threading
import time

ESP32_IP = "172.20.10.14"   # change if needed
ESP32_PORT = 4210
TIMEOUT_SEC = 2.0

HEARTBEAT_PERIOD_SEC = 1.0
PID_FILE = "/tmp/valve_client_heartbeat.pid"


def send_command(command: str, expect_reply: bool = True):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(TIMEOUT_SEC)

    try:
        sock.sendto(command.encode("utf-8"), (ESP32_IP, ESP32_PORT))
        print(f"Sent: {command}")

        if expect_reply:
            data, addr = sock.recvfrom(1024)
            print(f"Reply from {addr}: {data.decode('utf-8', errors='ignore')}")
    except socket.timeout:
        if expect_reply:
            print("No reply received (timeout).")
    finally:
        sock.close()


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def read_pid_file():
    if not os.path.exists(PID_FILE):
        return None
    try:
        with open(PID_FILE, "r") as f:
            return int(f.read().strip())
    except Exception:
        return None


def write_pid_file():
    with open(PID_FILE, "w") as f:
        f.write(str(os.getpid()))


def remove_pid_file():
    if os.path.exists(PID_FILE):
        try:
            os.remove(PID_FILE)
        except OSError:
            pass


def stop_existing_heartbeat_owner():
    pid = read_pid_file()
    if pid is None:
        return

    if not pid_alive(pid):
        remove_pid_file()
        return

    if pid != os.getpid():
        try:
            os.kill(pid, signal.SIGTERM)
            print(f"Stopped heartbeat process {pid}")
        except ProcessLookupError:
            pass
        time.sleep(0.2)
        remove_pid_file()


def heartbeat_loop(stop_event: threading.Event):
    while not stop_event.wait(HEARTBEAT_PERIOD_SEC):
        send_command("hb", expect_reply=False)


def run_start_mode():
    old_pid = read_pid_file()
    if old_pid is not None:
        if pid_alive(old_pid):
            print(f"Heartbeat process already running with PID {old_pid}.")
            print("Stop it first with: python3 valve_client.py x  or  python3 valve_client.py k")
            return
        else:
            remove_pid_file()

    # send start spray once
    send_command("s", expect_reply=True)

    stop_event = threading.Event()

    def handle_exit(signum, frame):
        print("\nStopping heartbeat. ESP32 will fail safe after watchdog timeout.")
        stop_event.set()

    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    write_pid_file()
    atexit.register(remove_pid_file)

    t = threading.Thread(target=heartbeat_loop, args=(stop_event,), daemon=True)
    t.start()

    print("Heartbeat running automatically every 1 second.")
    print("Kill this process or press Ctrl+C to stop heartbeat.")
    print("If heartbeat stops, ESP32 should shut down after ~5 seconds.")

    try:
        while not stop_event.is_set():
            time.sleep(0.5)
    finally:
        remove_pid_file()


def main():
    if len(sys.argv) != 2:
        print("Usage:")
        print("  python3 valve_client.py s")
        print("  python3 valve_client.py x")
        print("  python3 valve_client.py k")
        print("  python3 valve_client.py status")
        return

    cmd = sys.argv[1].strip().lower()

    if cmd == "s":
        run_start_mode()
        return

    if cmd == "x":
        send_command("x", expect_reply=True)
        stop_existing_heartbeat_owner()
        return

    if cmd == "k":
        send_command("k", expect_reply=True)
        stop_existing_heartbeat_owner()
        return

    if cmd == "status":
        send_command("status", expect_reply=True)
        return

    print("Unknown command. Use: s, x, k, status")


if __name__ == "__main__":
    main()
