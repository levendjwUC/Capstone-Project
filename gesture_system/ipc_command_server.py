# file: ipc_command_server.py
# 3-Gesture minimal server.
# Accepts: forward / left / ascend / hover

import socket
import threading
import time
from typing import Optional

HOST = "127.0.0.1"
PORT = 5007

VALID_COMMANDS = {"forward", "left", "ascend", "hover"}


class CommandServer:
    def __init__(self, host=HOST, port=PORT):
        self.host = host
        self.port = port
        self.server_socket: Optional[socket.socket] = None
        self.client_socket: Optional[socket.socket] = None
        self.client_address = None

        self.running = False
        self.latest_command: Optional[str] = None
        self._lock = threading.Lock()

    def start(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)

        self.running = True
        print(f"[SERVER] Listening on {self.host}:{self.port}...")

        thread = threading.Thread(target=self._accept_loop, daemon=True)
        thread.start()

    def _accept_loop(self):
        while self.running:
            try:
                sock, addr = self.server_socket.accept()
                print(f"[SERVER] Client connected from {addr}")
                with self._lock:
                    self.client_socket = sock
                    self.client_address = addr
                self._client_loop(sock)
            except OSError:
                break
            except Exception as e:
                print(f"[SERVER] Accept error: {e}")
                time.sleep(1.0)

    def _client_loop(self, sock):
        buffer = b""
        try:
            while self.running:
                data = sock.recv(1024)
                if not data:
                    print("[SERVER] Client disconnected.")
                    break

                buffer += data
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    cmd = line.decode().strip()
                    if cmd:
                        self._handle_command(cmd)
        except Exception as e:
            print(f"[SERVER] Client loop error: {e}")
        finally:
            sock.close()
            with self._lock:
                if sock is self.client_socket:
                    self.client_socket = None
                    self.client_address = None

    def _handle_command(self, cmd):
        if cmd not in VALID_COMMANDS:
            print(f"[SERVER] Ignored unknown command: {cmd}")
            return

        with self._lock:
            self.latest_command = cmd

        print(f"[SERVER] Command: {cmd}")

    def get_latest_command(self):
        with self._lock:
            return self.latest_command

    def stop(self):
        self.running = False
        if self.client_socket:
            try: self.client_socket.close()
            except: pass
        if self.server_socket:
            try: self.server_socket.close()
            except: pass
        print("[SERVER] Stopped.")


def main():
    server = CommandServer()
    server.start()

    print("[SERVER] Accepting 3-gesture commands… (Ctrl+C to quit)\n")

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[SERVER] Stopping…")
    finally:
        server.stop()


if __name__ == "__main__":
    main()
