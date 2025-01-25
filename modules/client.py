import socket
import cv2
import os
import numpy as np
from enum import Enum

GOSTREAMER_IS_RUNNING = False
START_PROCESSING = None

## Temporary default locations. Will be dynamically changed in the future.
SOURCE_FILE = "output/server/source/source.jpg"
TARGET_FILE = "output/server/target/target.jpg"
OUTPUT_FILE = "output/server/swapped/swapped.jpg"


class GoClient:
    def __init__(self, conn: socket):
        self.conn = conn
        self.webcam_handler = self.WebcamCapturer(self.conn)
        self.file_handler= self.FileHandler(self.conn)
        self.running = self.webcam_handler.running
        self.command = None
    def setCommand(self, command: str):
        if command == Commands.SEND_SOURCE.name or command == Commands.SEND_TARGET.name or command == Commands.REQUEST_FILE.name or command == Commands.START_FRAMES.name or command == Commands.STOP_FRAMES.name or command == Commands.EXIT.name:
            self.command = command.name
    class WebcamCapturer:
        def __init__(self, conn:  socket):
            self.conn = conn
            self.data_buffer = b""
            self.running = False
    
        def start(self, width, height, fps):
            self.running = True
            print(f"Frame capture started with resolution {width}x{height} at {fps} FPS.")
            return True
    
        def read(self):
            while self.running:
                chunk = self.conn.recv(4096)
                if not chunk:
                    return None  # Connection closed
                
                self.data_buffer += chunk
                start_idx = self.data_buffer.find(b"\xff\xd8")
                end_idx = self.data_buffer.find(b"\xff\xd9")
    
                if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
                    frame_data = self.data_buffer[start_idx:end_idx + 2]
                    self.data_buffer = self.data_buffer[end_idx + 2:]
                    return cv2.imdecode(np.frombuffer(frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)
    
        def release(self):
            self.running = False
            self.conn.close()
    
    class FileHandler:
        def __init__(self, conn: socket):
            self.conn = conn

        def receive(self, output_path):
            with open(output_path, 'wb') as file:
                while True:
                    data = self.conn.recv(4096)
                    if not data:
                        break
                    file.write(data)
            print(f"File received and saved to {output_path}")


        def send(self, file_path):
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                return

            with open(file_path, 'rb') as file:
                while (chunk := file.read(4096)):
                    self.conn.sendall(chunk)
            print(f"File sent: {file_path}")


class Commands(Enum):
    SEND_SOURCE = "SEND_SOURCE"
    SEND_TARGET = "SEND_TARGET"
    REQUEST_FILE = "REQUEST_FILE"
    START_FRAMES = "START_FRAMES"
    STOP_FRAMES = "STOP_FRAMES"
    EXIT = "EXIT"
    
    @staticmethod
    def read_command(client: GoClient):
        command = client.conn.recv(1024).decode('utf-8').strip()
        try:
            
            client.setCommand(Commands[command].name)
        except KeyError:
            raise ValueError(f"Invalid command: {command}")
        
  

def ClientHandler(host = '0.0.0.0',port = 5000):
    global GOSTREAMER_IS_RUNNING, SOURCE_FILE, TARGET_FILE, OUTPUT_FILE, START_PROCESSING

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(1)

    print(f"Receiver is listening on {host}:{port}...")

    conn, addr = server_socket.accept()
    print(f"Connection established with {addr}")

    client = GoClient(conn)
    try:
        received_source = False
        received_target = False
        while GOSTREAMER_IS_RUNNING:

            # Wait for a command from the client
            Commands.read_command(client)
            
            if client.command == Commands.SEND_SOURCE.name:
                print("Preparing to send a file...")
                client.file_handler.send(SOURCE_FILE)
                received_source = True
            elif client.command == Commands.SEND_TARGET.name:
                print("Preparing to send a file...")
                client.file_handler.send(TARGET_FILE)
                received_target = True
            elif client.command == Commands.REQUEST_FILE.name:
                print("Client requested a file...")
                client.file_handler.receive(OUTPUT_FILE)
            elif client.command == Commands.START_FRAMES.name:
                received_target = True
                print("Starting frame capture...")
                client.webcam_handler.start(width=640, height=480, fps=30)  # Example resolution/FPS
                while client.webcam_handler.running:
                    frame = client.webcam_handler.read()
                    if frame is None:
                        break
                    
                client.webcam_handler.release()
            elif client.command == Commands.STOP_FRAMES.name:
                print("Stopping frame capture...")
                client.webcam_handler.release()
            elif client.command == Commands.EXIT.name:
                print("Exiting...")
                GOSTREAMER_IS_RUNNING = False
            elif (received_target and received_source) or client.webcam_handler.running:
                START_PROCESSING = True
            else:
                START_PROCESSING = False
                continue
    except Exception as e:
        print(f"Error in goStreamer: {e}")
    finally:
        conn.close()
        server_socket.close()
        cv2.destroyAllWindows()
