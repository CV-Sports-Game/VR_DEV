"""
Python-Unity Bridge for VR Sports Analysis
==========================================
This module provides real-time communication between Python AI models and Unity VR.

Communication Methods:
1. TCP Socket Server (Recommended for real-time)
2. File-based Communication (For testing)
3. HTTP REST API (For web-based integration)

Usage:
    # Start the bridge server
    python unity_bridge.py --mode socket --port 8888
    
    # Or use file-based communication
    python unity_bridge.py --mode file
    
    # Or start HTTP server
    python unity_bridge.py --mode http --port 5000
"""

import os
import json
import socket
import threading
import time
import argparse
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import numpy as np
import cv2
from analyze_video import SportsAnalyzer

class UnityBridge:
    """Bridge class for communicating with Unity VR application."""
    
    def __init__(self, analyzer=None):
        self.analyzer = analyzer or SportsAnalyzer()
        self.running = False
        self.clients = []
        
    def process_pose_data(self, pose_landmarks_json):
        """
        Process pose landmarks from Unity and return analysis results.
        
        Args:
            pose_landmarks_json: JSON string with pose landmarks from Unity
                                Format: {"landmarks": [[x, y, z], ...], "timestamp": float}
        
        Returns:
            dict: Analysis results with pose classification, confidence, and feedback
        """
        try:
            data = json.loads(pose_landmarks_json)
            landmarks = np.array(data.get('landmarks', []), dtype=np.float32)
            timestamp = data.get('timestamp', time.time())
            
            # Convert landmarks to 99-dimensional vector (33 landmarks × 3 coords)
            if len(landmarks) == 33:
                pose_vector = landmarks.flatten()  # Already 99 dimensions
            elif len(landmarks) == 0:
                pose_vector = np.zeros(99, dtype=np.float32)
            else:
                # Handle different formats
                pose_vector = np.zeros(99, dtype=np.float32)
                if landmarks.shape[0] >= 33:
                    pose_vector = landmarks[:33].flatten()[:99]
            
            # Add to sequence buffer for transformer analysis
            self.analyzer.pose_sequence_buffer.append(pose_vector)
            
            # Analyze using sequence model if available
            sequence_result = None
            if self.analyzer.pose_model and len(self.analyzer.pose_sequence_buffer) >= self.analyzer.sequence_len:
                sequence_result = self.analyzer.analyze_pose_sequence()
            
            # Create a dummy frame for CNN analysis (optional - Unity can send frame too)
            # For now, we'll focus on sequence analysis
            
            result = {
                'timestamp': timestamp,
                'success': True,
                'sequence_analysis': sequence_result,
                'buffer_size': len(self.analyzer.pose_sequence_buffer),
                'ready_for_sequence': len(self.analyzer.pose_sequence_buffer) >= self.analyzer.sequence_len
            }
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'timestamp': time.time()
            }
    
    def process_frame_image(self, image_data, image_format='jpg'):
        """
        Process image frame from Unity and return analysis results.
        
        Args:
            image_data: Base64 encoded image or numpy array
            image_format: Format of the image ('jpg', 'png', etc.)
        
        Returns:
            dict: Analysis results with pose classification
        """
        try:
            # Decode image if base64
            if isinstance(image_data, str):
                import base64
                image_bytes = base64.b64decode(image_data)
                nparr = np.frombuffer(image_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            else:
                frame = image_data
            
            # Save temporary image for CNN analysis
            temp_path = f"temp_unity_frame_{int(time.time())}.jpg"
            cv2.imwrite(temp_path, frame)
            
            # Analyze using CNN
            analysis = self.analyzer.analyze_image(temp_path)
            
            # Extract pose landmarks for sequence analysis
            pose_vector = self.analyzer.extract_pose_landmarks_to_vector(frame)
            self.analyzer.pose_sequence_buffer.append(pose_vector)
            
            # Sequence analysis
            sequence_result = None
            if self.analyzer.pose_model and len(self.analyzer.pose_sequence_buffer) >= self.analyzer.sequence_len:
                sequence_result = self.analyzer.analyze_pose_sequence()
            
            # Combine results
            result = {
                'success': True,
                'timestamp': time.time(),
                'frame_analysis': analysis,
                'sequence_analysis': sequence_result,
                'combined_confidence': (
                    analysis.get('confidence', 0) + 
                    (sequence_result.get('confidence', 0) if sequence_result else 0)
                ) / (2 if sequence_result else 1)
            }
            
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'timestamp': time.time()
            }


class SocketBridgeServer:
    """TCP Socket server for real-time Unity communication."""
    
    def __init__(self, bridge, host='localhost', port=8888):
        self.bridge = bridge
        self.host = host
        self.port = port
        self.socket = None
        self.running = False
    
    def handle_client(self, client_socket, address):
        """Handle individual client connection."""
        print(f"📡 Connected to Unity client: {address}")
        try:
            while self.running:
                # Receive data from Unity
                data = client_socket.recv(4096)
                if not data:
                    break
                
                try:
                    message = json.loads(data.decode('utf-8'))
                    message_type = message.get('type')
                    
                    if message_type == 'pose_landmarks':
                        # Process pose landmarks
                        result = self.bridge.process_pose_data(message.get('data', '{}'))
                        response = json.dumps(result)
                        client_socket.sendall(response.encode('utf-8'))
                    
                    elif message_type == 'frame':
                        # Process image frame
                        result = self.bridge.process_frame_image(
                            message.get('data'),
                            message.get('format', 'jpg')
                        )
                        response = json.dumps(result)
                        client_socket.sendall(response.encode('utf-8'))
                    
                    elif message_type == 'ping':
                        # Health check
                        response = json.dumps({'status': 'ok', 'timestamp': time.time()})
                        client_socket.sendall(response.encode('utf-8'))
                    
                    elif message_type == 'reset':
                        # Reset sequence buffer
                        self.bridge.analyzer.pose_sequence_buffer.clear()
                        response = json.dumps({'status': 'reset', 'timestamp': time.time()})
                        client_socket.sendall(response.encode('utf-8'))
                    
                    else:
                        response = json.dumps({'error': 'Unknown message type', 'type': message_type})
                        client_socket.sendall(response.encode('utf-8'))
                        
                except json.JSONDecodeError:
                    response = json.dumps({'error': 'Invalid JSON'})
                    client_socket.sendall(response.encode('utf-8'))
                    
        except Exception as e:
            print(f"❌ Error handling client {address}: {e}")
        finally:
            client_socket.close()
            print(f"📴 Disconnected from Unity client: {address}")
    
    def start(self):
        """Start the socket server."""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((self.host, self.port))
        self.socket.listen(5)
        self.running = True
        
        print(f"🚀 Unity Bridge Server started on {self.host}:{self.port}")
        print("   Waiting for Unity client connections...")
        
        try:
            while self.running:
                client_socket, address = self.socket.accept()
                client_thread = threading.Thread(
                    target=self.handle_client,
                    args=(client_socket, address)
                )
                client_thread.daemon = True
                client_thread.start()
        except KeyboardInterrupt:
            print("\n🛑 Shutting down server...")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the socket server."""
        self.running = False
        if self.socket:
            self.socket.close()
        print("✅ Server stopped")


class FileBridgeServer:
    """File-based communication for testing Unity integration."""
    
    def __init__(self, bridge, input_file='unity_input.json', output_file='unity_output.json'):
        self.bridge = bridge
        self.input_file = input_file
        self.output_file = output_file
        self.running = False
    
    def start(self):
        """Start file-based bridge monitoring."""
        self.running = True
        print(f"📁 File-based Bridge started")
        print(f"   Watching: {self.input_file}")
        print(f"   Output: {self.output_file}")
        print("   Press Ctrl+C to stop")
        
        try:
            while self.running:
                if os.path.exists(self.input_file):
                    try:
                        with open(self.input_file, 'r') as f:
                            message = json.load(f)
                        
                        message_type = message.get('type')
                        result = None
                        
                        if message_type == 'pose_landmarks':
                            result = self.bridge.process_pose_data(message.get('data', '{}'))
                        elif message_type == 'frame':
                            result = self.bridge.process_frame_image(
                                message.get('data'),
                                message.get('format', 'jpg')
                            )
                        elif message_type == 'reset':
                            self.bridge.analyzer.pose_sequence_buffer.clear()
                            result = {'status': 'reset', 'timestamp': time.time()}
                        
                        if result:
                            with open(self.output_file, 'w') as f:
                                json.dump(result, f, indent=2)
                            
                            # Remove input file after processing
                            os.remove(self.input_file)
                            print(f"✅ Processed {message_type} request")
                    
                    except Exception as e:
                        print(f"❌ Error processing file: {e}")
                
                time.sleep(0.1)  # Check every 100ms
                
        except KeyboardInterrupt:
            print("\n🛑 Shutting down file bridge...")
        finally:
            self.running = False
            print("✅ File bridge stopped")


class HTTPBridgeHandler(BaseHTTPRequestHandler):
    """HTTP request handler for REST API bridge."""
    
    bridge = None
    
    def do_POST(self):
        """Handle POST requests from Unity."""
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        
        try:
            message = json.loads(post_data.decode('utf-8'))
            message_type = message.get('type')
            result = None
            
            if message_type == 'pose_landmarks':
                result = self.bridge.process_pose_data(message.get('data', '{}'))
            elif message_type == 'frame':
                result = self.bridge.process_frame_image(
                    message.get('data'),
                    message.get('format', 'jpg')
                )
            elif message_type == 'reset':
                self.bridge.analyzer.pose_sequence_buffer.clear()
                result = {'status': 'reset', 'timestamp': time.time()}
            
            if result:
                response = json.dumps(result)
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(response.encode('utf-8'))
            else:
                self.send_response(400)
                self.end_headers()
                
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            error_response = json.dumps({'error': str(e)})
            self.wfile.write(error_response.encode('utf-8'))
    
    def do_GET(self):
        """Handle GET requests (health check)."""
        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = json.dumps({
                'status': 'ok',
                'timestamp': time.time(),
                'buffer_size': len(self.bridge.analyzer.pose_sequence_buffer),
                'ready': len(self.bridge.analyzer.pose_sequence_buffer) >= self.bridge.analyzer.sequence_len
            })
            self.wfile.write(response.encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        """Suppress default logging."""
        pass


def main():
    """Main entry point for Unity bridge."""
    parser = argparse.ArgumentParser(description='Python-Unity Bridge for VR Sports Analysis')
    parser.add_argument('--mode', type=str, choices=['socket', 'file', 'http'], 
                       default='socket', help='Communication mode (default: socket)')
    parser.add_argument('--host', type=str, default='localhost', 
                       help='Host address (default: localhost)')
    parser.add_argument('--port', type=int, default=8888, 
                       help='Port number (default: 8888)')
    parser.add_argument('--input-file', type=str, default='unity_input.json',
                       help='Input file for file mode (default: unity_input.json)')
    parser.add_argument('--output-file', type=str, default='unity_output.json',
                       help='Output file for file mode (default: unity_output.json)')
    parser.add_argument('--image-model', type=str, default='image_model.pth',
                       help='Path to CNN image model')
    parser.add_argument('--pose-model', type=str, default='pose_model.pth',
                       help='Path to PoseTransformer model')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    print("🤖 Initializing AI models...")
    analyzer = SportsAnalyzer(
        image_model_path=args.image_model,
        pose_model_path=args.pose_model,
        use_sequence_model=True
    )
    
    # Initialize bridge
    bridge = UnityBridge(analyzer)
    
    # Start server based on mode
    if args.mode == 'socket':
        server = SocketBridgeServer(bridge, host=args.host, port=args.port)
        server.start()
    
    elif args.mode == 'file':
        server = FileBridgeServer(bridge, 
                                  input_file=args.input_file,
                                  output_file=args.output_file)
        server.start()
    
    elif args.mode == 'http':
        HTTPBridgeHandler.bridge = bridge
        http_server = HTTPServer((args.host, args.port), HTTPBridgeHandler)
        print(f"🌐 HTTP Bridge Server started on http://{args.host}:{args.port}")
        print("   Endpoints:")
        print("     POST / - Send pose landmarks or frame data")
        print("     GET /health - Health check")
        try:
            http_server.serve_forever()
        except KeyboardInterrupt:
            print("\n🛑 Shutting down HTTP server...")
            http_server.shutdown()
            print("✅ HTTP server stopped")


if __name__ == "__main__":
    main()

