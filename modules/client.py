import socket
import cv2
import os
import numpy as np
from enum import Enum
import modules.globals
import json
## Temporary default locations. Will be dynamically changed in the future.
SOURCE_DIR = "output/server/source/"
TARGET_DIR = "output/server/target/"
OUTPUT_DIR = "output/server/swapped/"

class response:
    def __init__(self, command, file_name, file_size):
        self.command = command
        self.file_name = file_name
        self.file_size = file_size
    def __repr__(self):
        return f"response(command={self.command}, file_name={self.file_name}, file_size={self.file_size})"
    
def response_to_struct(d):
    return response(**d)

class Commands(Enum):
    SEND_SOURCE = "SEND_SOURCE"
    SEND_TARGET = "SEND_TARGET"
    REQUEST_FILE = "REQUEST_FILE"
    START_FRAMES = "START_FRAMES"
    STOP_FRAMES = "STOP_FRAMES"
    EXIT = "EXIT"
    
    @staticmethod
    def getResponse(client_conn):
        data = client_conn.recv(1024)
        try:
            if not data:
                raise ConnectionError("Client disconnected or sent empty data.")
            decoded = data.decode('utf-8', errors='ignore').strip()
            command = decoded.split("\n")[0]
            if not command:
                raise ValueError("Received an empty command string.")
            split_data = decoded.split("\n", 1)
            additional_data = split_data[1].strip() if len(split_data) > 1 else None  # Remaining data

            resp=response_to_struct(json.loads(f"{split_data[0]}".replace("0089","",1)))

            return resp, additional_data
        except:
            pass
        return None, data


class WebcamCapturer:
    def __init__(self, conn: socket.socket):
        self.conn = conn
        self.data_buffer = b""
        self.running = False

    def start(self, width=640, height=480, fps=30):
        if self.running:
            return
        self.running = True
        print(f"Frame capture started with resolution {width}x{height} at {fps} FPS.")

    def read(self):
        try:
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
        except Exception as e:
            print(f"Error reading frame: {e}")
            return None

    def release(self):
        self.running = False
        print("Frame capture stopped.")

        
class FileHandler:
    def __init__(self, conn: socket.socket):
        self.conn = conn

    def recieve_from_client(self,response: response, output_path):
        print("Receiving file", output_path)
        try:
            buffer = b""
            while len(buffer) < response.file_size:
                data = self.conn.recv(4096)
                if not data:
                    raise RuntimeError("Connection closed before receiving full file")
                buffer += data

            # Save the file
            with open(output_path, "wb") as file:
                file.write(buffer)

            print("File saved to", output_path)
        except Exception as e:
            print(f"Error receiving file: {e}")

    
    def send_to_client(self, file_path):
        try:
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                return
    
            # Get the file size
            file_size = os.path.getsize(file_path)
            file_name = os.path.basename(file_path)
    
            # Prepare header as JSON
            header = {
                "command": "SEND_FILE",
                "file_name": file_name,
                "file_size": file_size
            }
            header_data = json.dumps(header).encode('utf-8')
    
            # Send header length and data
            self.conn.sendall(header_data)
    
            # Send the file in chunks
            with open(file_path, 'rb') as file:
                while (chunk := file.read(4096)):
                    self.conn.sendall(chunk)
            print(f"File sent: {file_path}")
        except Exception as e:
            print(f"Error sending file: {e}")


class GoClient:
    def __init__(self):
        self.conn = None
        self.webcam_handler = None
        self.file_handler = None
        self.running = None
        self.ready = None
    def setConn(self,conn: socket.socket):
        self.conn = conn
        self.file_handler = FileHandler(conn)
        self.webcam_handler = WebcamCapturer(self.conn)
        self.webcam_handler.running

    def sendMsg(self, msg: str):
        self.conn.sendall(msg.encode('utf-8'))
    def request_file(self,size, file_path):
        count = 0
        ok = False
        while not ok:
            ok = self.file_handler.recieve_from_client(size, file_path)
            if not ok:
                print("Error receiving source file. Retrying...")
                self.sendMsg("RETRY")
                count += 1
            if count > 3:
                print("Failed to receive source file after 3 attempts. Exiting...")
                break
        self.sendMsg("DONE")
        return ok


client = GoClient()

def handle_client(conn, addr):
    have_target = None
    have_source = None
    client.ready = False
    print(f"Connection established with {addr}")
    client.setConn(conn)
    while not client.ready:
        command = None
        response, data = Commands.getResponse(client.conn)
        if response:
            command = Commands[response.command]
        elif "REQUEST_FILE" in str(data):
            command = Commands["REQUEST_FILE"]
    
        if command is None:
            print(f"invalid command")
            return
        if command == Commands.SEND_SOURCE:
            client.file_handler.recieve_from_client(response,os.path.join(SOURCE_DIR,os.path.basename(response.file_name)))
            client.sendDone()
            have_source = True
        elif command == Commands.SEND_TARGET:
            client.file_handler.recieve_from_client(response,os.path.join(TARGET_DIR,os.path.basename(response.file_name)))
            client.sendDone()
            have_target = True
        elif command == Commands.REQUEST_FILE:
            client.file_handler.send_to_client(os.path.join(OUTPUT_DIR,os.path.basename(modules.globals.output_path)))
        elif command == Commands.START_FRAMES:
            have_source = True
            client.webcam_handler.start(width=640, height=480, fps=30)
            while client.webcam_handler.running and client.ready:
                frame = client.webcam_handler.read()
                if frame is None:
                    return
                # Process frame here if needed
            client.webcam_handler.release()
        elif command == Commands.STOP_FRAMES:
            client.webcam_handler.release()
        elif command == Commands.EXIT:
            print("Exiting...")
            return
        if have_source and have_target:
            client.ready = True


def start_server(host="0.0.0.0", port=5000):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(5)
    print(f"Server listening on {host}:{port}...")

    try:
        while True:
            conn, addr = server_socket.accept()
            handle_client(conn, addr)
    except KeyboardInterrupt:
        print("Server shutting down...")
    finally:
        server_socket.close()
        cv2.destroyAllWindows()

