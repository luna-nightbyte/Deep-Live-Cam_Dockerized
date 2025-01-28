import socket
import cv2
import os
import numpy as np
from enum import Enum
import modules.globals
import json
from time import sleep

DEFAULT_SOURCE_FOLDER="output/source"
DEFAULT_TARGET_FOLDER="output/target"
DEFAULT_OUTPUT_FOLDER="output/swapped"

class CloudServer:
        
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
    
        
    
        def recieve_from_client(self,size, output_path):
            print("Receiving file", output_path)
            try:
                buffer = b""
                while len(buffer) < size:
                    data = self.conn.recv(4096)
                    if not data:
                        raise RuntimeError("Connection closed before receiving full file")
                    buffer += data
    
                # Save the file
                folder_path = os.path.dirname(output_path)
                os.makedirs(folder_path, exist_ok=True)
                with open(output_path, "wb") as file:
                    file.write(buffer)
    
                print("File saved to", output_path)
                return True
            except Exception as e:
                print(f"Error receiving file: {e}")
                return False
                
    
        
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
    
    def __init__(self):
        self.conn = None
        self.webcam_handler = None
        self.file_handler = None
        self.running = None
        self.ready = None
        self.have_source = None
        self.have_target = None

    def setConn(self,conn: socket.socket):
        self.conn = conn
        self.file_handler = self.FileHandler(conn)
        self.webcam_handler = self.WebcamCapturer(self.conn)
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
    def lock(self):
        """
        Resett variables will "lock" the state until all files and args have been recceived. Perfect for a loop where we wait for new commands
        """
        self.ready = False
        self.have_source = False
        self.have_target = False

    class response:
        def __init__(self, client_ip, client_port, command: str, file_name: str, file_size: int):
            self.client_ip = client_ip
            self.client_port = client_port
            self.command = command
            self.file_name = file_name
            self.file_size = file_size
        def __repr__(self):
            return f"response(client_ip={self.client_ip}, client_port={self.client_port}, command={self.command}, file_name={self.file_name}, file_size={self.file_size})"
    
    def response_to_struct(self,d):
        return self.response(**d)


server = CloudServer()


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

            resp=server.response_to_struct(json.loads(f"{split_data[0]}".replace("0089","",1)))

            return resp, additional_data
        except:
            pass
        return None, data


def handle_client(conn, addr) -> bool:
    print(f"Connection established with {addr}")
    server.setConn(conn)
    try:
        while not server.ready:
            command = None
            response, data = Commands.getResponse(server.conn)
            if response:
                command = Commands[response.command]
            elif "REQUEST_FILE" in str(data):
                command = Commands["REQUEST_FILE"]

            if command is None:
                print(f"invalid command")
                continue
            if command == Commands.SEND_SOURCE:
                ok = server.request_file(response.file_size, os.path.join(resolve_dir(modules.globals.source_path),os.path.basename(response.file_name)))
                if not ok:
                    print("Error receiving source file!")
                else:
                    server.have_source = True
            elif command == Commands.SEND_TARGET:
                ok = server.request_file(response.file_size, os.path.join(resolve_dir(modules.globals.target_path),os.path.basename(response.file_name)))
                if not ok:
                    print("Error receiving target file!")
                else:
                    server.have_source = True

            elif command == Commands.REQUEST_FILE:
                server.file_handler.send_to_client(os.path.join(resolve_dir(modules.globals.output_path),os.path.basename(response.file_name)))
            elif command == Commands.START_FRAMES:
                server.have_source = True
                server.webcam_handler.start(width=640, height=480, fps=30)
                while server.webcam_handler.running and server.ready:
                    frame = server.webcam_handler.read()
                    if frame is None:
                        return True
                    # Process frame here if needed
                server.webcam_handler.release()
            elif command == Commands.STOP_FRAMES:
                server.webcam_handler.release()
            elif command == Commands.EXIT:
                print("Exiting...")
                return False
            if server.have_source and server.have_target:
                server.ready = True
        return True
    except KeyboardInterrupt:
            print("Server stopping")
            return False

def start_listener(host="0.0.0.0", port=5000):
    run = True
    closed = False
    while run:
        # Reset input status
        server.lock()
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind((host, port))
        server_socket.listen(5)
        print(f"Server listening on {host}:{port}...")

        try:
            while True:
                conn, addr = server_socket.accept()
                run = handle_client(conn, addr)
                if run:
                    continue
                else:
                    print("Server Stopping")
                    break
        except KeyboardInterrupt:
            print("Server Stopping")
        finally:
            print("Closing socket...")
            server_socket.close()
            closed=True
            continue
    if not closed:
        server_socket.close()

def wait_for_ready_signal():
    server.ready = False
    if modules.globals.server_only:
            print("Waiting for client files..")

    while modules.globals.server_only and not server.ready:
        sleep(60)
        print("Still waiting for client files..")

def resolve_dir(path: str):
    if os.path.isdir(path):
        return path
    else:
        return os.path.dirname(path)