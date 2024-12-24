import cv2
import numpy as np
import datetime
import time
import os
import requests
import threading
import configparser
from pathlib import Path
from flask import Flask, Response, render_template
import psutil
import json
from io import BytesIO

class SecurityCamera:
    def __init__(self):
        # Initialize configuration
        self.config = configparser.ConfigParser()
        self.config.read('config.ini')
        
        # Telegram settings
        self.telegram_token = self.config['Telegram']['token']
        self.telegram_chat_id = self.config['Telegram']['chat_id']
        
        # System state
        self.is_armed = True
        self.start_time = datetime.datetime.now()
        self.events_today = 0
        self.last_event_time = None
        
        # Camera settings
        self.camera = cv2.VideoCapture(0)
        self.frame_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Motion detection settings
        self.min_area = 500
        self.motion_detected = False
        self.last_notification_time = 0
        self.notification_cooldown = 60
        
        # Create output directory
        self.output_dir = Path('security_footage')
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=True)
        
        # Flask app for web streaming
        self.app = Flask(__name__)
        self.frame = None
        self.lock = threading.Lock()
        
        # Start Telegram command polling
        threading.Thread(target=self._poll_telegram_commands, daemon=True).start()

    def _poll_telegram_commands(self):
        """Poll for and process Telegram commands"""
        last_update_id = 0
        while True:
            try:
                url = f'https://api.telegram.org/bot{self.telegram_token}/getUpdates'
                params = {'offset': last_update_id + 1, 'timeout': 30}
                response = requests.get(url, params=params).json()
                
                if response['ok'] and response['result']:
                    for update in response['result']:
                        last_update_id = update['update_id']
                        if 'message' in update and 'text' in update['message']:
                            self._handle_telegram_command(update['message']['text'])
                            
            except Exception as e:
                print(f"Error polling Telegram commands: {e}")
            time.sleep(1)

    def _handle_telegram_command(self, command):
        """Handle Telegram commands"""
        if command == '/arm':
            self.is_armed = True
            self.send_telegram_message("System armed! Motion detection active.")
            
        elif command == '/disarm':
            self.is_armed = False
            self.send_telegram_message("System disarmed! Motion detection disabled.")
            
        elif command == '/photo':
            self._send_current_photo()
            
        elif command == '/status':
            self._send_status()

    def _send_status(self):
        """Send system status information"""
        uptime = datetime.datetime.now() - self.start_time
        status_msg = (
            "🎥 *Security Camera Status*\n"
            f"🔐 Motion Detection: {'Armed' if self.is_armed else 'Disarmed'}\n"
            f"⏱ Uptime: {str(uptime).split('.')[0]}\n"
            f"🔍 Events Today: {self.events_today}\n"
            f"🕒 Last Event: {self.last_event_time.strftime('%H:%M:%S') if self.last_event_time else 'None'}\n"
            f"📹 Camera: {'Online' if self.camera.isOpened() else 'Offline'}\n"
            f"💾 Storage Free: {self._get_free_space()}GB\n"
            f"🔄 CPU Usage: {psutil.cpu_percent()}%\n"
            f"📊 Memory Usage: {psutil.virtual_memory().percent}%"
        )
        self.send_telegram_message(status_msg, parse_mode='Markdown')

    def _get_free_space(self):
        """Get free space in GB"""
        usage = psutil.disk_usage(self.output_dir)
        return round(usage.free / (1024 * 1024 * 1024), 2)

    def _send_current_photo(self):
        """Capture and send current photo"""
        try:
            with self.lock:
                if self.frame is not None:
                    _, buffer = cv2.imencode('.jpg', self.frame)
                    bio = BytesIO(buffer)
                    bio.seek(0)
                    self._send_telegram_photo(bio)
        except Exception as e:
            self.send_telegram_message(f"Error capturing photo: {str(e)}")

    def send_telegram_message(self, message, parse_mode=None):
        """Send text message via Telegram"""
        try:
            url = f'https://api.telegram.org/bot{self.telegram_token}/sendMessage'
            data = {
                'chat_id': self.telegram_chat_id,
                'text': message
            }
            if parse_mode:
                data['parse_mode'] = parse_mode
            requests.post(url, data=data)
        except Exception as e:
            print(f"Error sending Telegram message: {e}")

    def _send_telegram_photo(self, photo_bio):
        """Send photo via Telegram"""
        try:
            url = f'https://api.telegram.org/bot{self.telegram_token}/sendPhoto'
            files = {'photo': ('image.jpg', photo_bio)}
            data = {'chat_id': self.telegram_chat_id}
            requests.post(url, data=data, files=files)
        except Exception as e:
            print(f"Error sending Telegram photo: {e}")

    def send_telegram_notification(self, image_path):
        """Send motion detection notification with photo"""
        try:
            message = f"Motion detected! {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            self.send_telegram_message(message)
            
            with open(image_path, 'rb') as photo:
                self._send_telegram_photo(photo)
                
            self.events_today += 1
            self.last_event_time = datetime.datetime.now()
            print("Telegram notification sent successfully")
        except Exception as e:
            print(f"Failed to send Telegram notification: {str(e)}")

    def start_camera(self):
        """Start the camera and motion detection in a separate thread"""
        threading.Thread(target=self._camera_loop, daemon=True).start()
        
    def _camera_loop(self):
        """Main camera loop for motion detection"""
        while True:
            ret, frame = self.camera.read()
            if not ret:
                break
                
            # Update frame for web streaming
            with self.lock:
                self.frame = frame.copy()
                
            # Skip motion detection if system is disarmed
            if not self.is_armed:
                continue
                
            # Motion detection
            fg_mask = self.bg_subtractor.apply(frame)
            fg_mask = cv2.threshold(fg_mask, 244, 255, cv2.THRESH_BINARY)[1]
            
            contours, _ = cv2.findContours(
                fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Check for motion
            motion_detected = False
            for contour in contours:
                if cv2.contourArea(contour) > self.min_area:
                    motion_detected = True
                    break
            
            # Handle motion detection with cooldown
            current_time = time.time()
            if motion_detected and not self.motion_detected and \
               (current_time - self.last_notification_time) > self.notification_cooldown:
                self.motion_detected = True
                self.last_notification_time = current_time
                self._handle_motion_detection(frame)
            elif not motion_detected:
                self.motion_detected = False
            
    def _handle_motion_detection(self, frame):
        """Handle motion detection event"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = self.output_dir / f"motion_{timestamp}.jpg"
        
        # Save image
        cv2.imwrite(str(image_path), frame)
        
        # Send notification
        self.send_telegram_notification(image_path)
        
    def generate_frames(self):
        """Generator function for web streaming"""
        while True:
            with self.lock:
                if self.frame is not None:
                    _, buffer = cv2.imencode('.jpg', self.frame)
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    def start_server(self, host='0.0.0.0', port=5000):
        """Start the Flask server for web streaming"""
        @self.app.route('/')
        def index():
            return render_template('index.html')
        
        @self.app.route('/video_feed')
        def video_feed():
            return Response(self.generate_frames(),
                          mimetype='multipart/x-mixed-replace; boundary=frame')
        
        self.app.run(host=host, port=port, debug=False)

if __name__ == "__main__":
    camera = SecurityCamera()
    camera.start_camera()
    camera.start_server()
