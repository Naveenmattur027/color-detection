import streamlit as st
import cv2
import numpy as np
import collections
from PIL import Image
import threading
import time
import queue

class MultiColorTracker:
    def __init__(self, buffer_size=64):
        """
        Initialize the multi-color tracker that detects all colors
        """
        self.buffer_size = buffer_size
        self.tracked_objects = {}  # Dictionary to store multiple object trails
        
        # Define comprehensive HSV color ranges
        self.color_ranges = {
            'Red': [([0, 50, 50], [10, 255, 255]), ([170, 50, 50], [180, 255, 255])],
            'Orange': [([10, 50, 50], [25, 255, 255])],
            'Yellow': [([25, 50, 50], [35, 255, 255])],
            'Green': [([35, 50, 50], [85, 255, 255])],
            'Cyan': [([85, 50, 50], [95, 255, 255])],
            'Blue': [([95, 50, 50], [125, 255, 255])],
            'Purple': [([125, 50, 50], [145, 255, 255])],
            'Pink': [([145, 50, 50], [170, 255, 255])]
        }
        
        # Color to BGR mapping for visualization
        self.color_bgr = {
            'Red': (0, 0, 255),
            'Orange': (0, 165, 255),
            'Yellow': (0, 255, 255),
            'Green': (0, 255, 0),
            'Cyan': (255, 255, 0),
            'Blue': (255, 0, 0),
            'Purple': (128, 0, 128),
            'Pink': (203, 192, 255)
        }
        
        # Camera properties
        self.cap = None
        self.frame_queue = queue.Queue(maxsize=2)
        self.camera_thread = None
        self.camera_running = False
        
    def start_camera(self, camera_index=0):
        """Start camera capture in separate thread"""
        if self.camera_running:
            return True
            
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            return False
            
        # Set camera properties for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        self.camera_running = True
        self.camera_thread = threading.Thread(target=self._camera_loop)
        self.camera_thread.daemon = True
        self.camera_thread.start()
        
        return True
    
    def _camera_loop(self):
        """Camera capture loop running in separate thread"""
        while self.camera_running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                if not self.frame_queue.full():
                    self.frame_queue.put(frame)
                else:
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put(frame)
                    except queue.Empty:
                        pass
            time.sleep(0.01)
    
    def stop_camera(self):
        """Stop camera capture"""
        self.camera_running = False
        if self.camera_thread:
            self.camera_thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
            self.cap = None
    
    def get_latest_frame(self):
        """Get the latest frame from camera"""
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None
    
    def create_color_mask(self, hsv_frame, color_name):
        """Create mask for a specific color"""
        masks = []
        
        for lower, upper in self.color_ranges[color_name]:
            mask = cv2.inRange(hsv_frame, np.array(lower), np.array(upper))
            masks.append(mask)
        
        # Combine all masks for this color
        final_mask = masks[0]
        for mask in masks[1:]:
            final_mask = cv2.bitwise_or(final_mask, mask)
        
        # Apply morphological operations
        kernel_size = st.session_state.get('kernel_size', 5)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
        final_mask = cv2.GaussianBlur(final_mask, (5, 5), 0)
        
        return final_mask
    
    def find_largest_contour(self, mask):
        """Find the largest contour in the mask"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, None
        
        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Only proceed if contour is large enough
        min_area = st.session_state.get('min_area', 300)
        if cv2.contourArea(largest_contour) < min_area:
            return None, None
        
        # Calculate center using moments
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            return center, largest_contour
        
        return None, None
    
    def detect_all_colors(self, frame):
        """Detect all colors in the frame and return detected objects"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        detected_objects = []
        
        for color_name in self.color_ranges.keys():
            # Create mask for this color
            mask = self.create_color_mask(hsv, color_name)
            
            # Find largest contour for this color
            center, contour = self.find_largest_contour(mask)
            
            if center is not None and contour is not None:
                area = cv2.contourArea(contour)
                detected_objects.append({
                    'color': color_name,
                    'center': center,
                    'contour': contour,
                    'area': area,
                    'bgr_color': self.color_bgr[color_name]
                })
        
        # Sort by area (largest first)
        detected_objects.sort(key=lambda x: x['area'], reverse=True)
        
        return detected_objects
    
    def update_tracking_trails(self, detected_objects):
        """Update tracking trails for all detected objects"""
        current_colors = set()
        
        for obj in detected_objects:
            color_name = obj['color']
            center = obj['center']
            current_colors.add(color_name)
            
            # Initialize trail if new color
            if color_name not in self.tracked_objects:
                self.tracked_objects[color_name] = collections.deque(maxlen=self.buffer_size)
            
            # Add current center to trail
            self.tracked_objects[color_name].appendleft(center)
        
        # Remove trails for colors not currently detected (optional - commented out to keep trails longer)
        # colors_to_remove = set(self.tracked_objects.keys()) - current_colors
        # for color in colors_to_remove:
        #     del self.tracked_objects[color]
    
    def draw_all_tracking_info(self, frame, detected_objects):
        """Draw tracking information for all detected objects"""
        self.update_tracking_trails(detected_objects)
        
        # Draw trails for all tracked colors
        trail_thickness = st.session_state.get('trail_thickness', 2)
        
        for color_name, points in self.tracked_objects.items():
            color_bgr = self.color_bgr.get(color_name, (255, 255, 255))
            
            for i in range(1, len(points)):
                if points[i - 1] is None or points[i] is None:
                    continue
                
                # Calculate thickness (thicker = more recent)
                thickness = int(np.sqrt(self.buffer_size / float(i + 1)) * trail_thickness)
                cv2.line(frame, points[i - 1], points[i], color_bgr, max(1, thickness))
        
        # Draw current detection info
        for i, obj in enumerate(detected_objects):
            color_name = obj['color']
            center = obj['center']
            contour = obj['contour']
            area = obj['area']
            color_bgr = obj['bgr_color']
            
            # Draw bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color_bgr, 2)
            
            # Draw center point
            cv2.circle(frame, center, 8, color_bgr, -1)
            cv2.circle(frame, center, 3, (255, 255, 255), -1)
            
            # Draw label
            label = f"{color_name}: {int(area)}px²"
            label_pos = (x, y - 10 if y > 20 else y + h + 20)
            
            # Draw label background
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (label_pos[0] - 2, label_pos[1] - label_h - 2), 
                         (label_pos[0] + label_w + 2, label_pos[1] + 2), color_bgr, -1)
            
            # Draw label text
            cv2.putText(frame, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw summary info
        cv2.putText(frame, f"Objects Detected: {len(detected_objects)}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if detected_objects:
            colors_detected = [obj['color'] for obj in detected_objects]
            colors_text = ", ".join(colors_detected[:3])  # Show first 3 colors
            if len(detected_objects) > 3:
                colors_text += f" +{len(detected_objects)-3} more"
            
            cv2.putText(frame, f"Colors: {colors_text}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return frame
    
    def process_frame(self, frame):
        """Process frame for multi-color detection and tracking"""
        if frame is None:
            return None, []
        
        # Detect all colors in frame
        detected_objects = self.detect_all_colors(frame)
        
        # Draw tracking information
        tracked_frame = self.draw_all_tracking_info(frame.copy(), detected_objects)
        
        return tracked_frame, detected_objects

def main():
    st.set_page_config(
        page_title="Multi-Color Tracking",
        page_icon="🌈",
        layout="wide"
    )
    
    st.title("🌈 Multi-Color Object Detection & Tracking")
    st.markdown("### Automatically detect and track all colored objects in real-time")
    
    # Initialize session state
    if 'tracker' not in st.session_state:
        st.session_state.tracker = MultiColorTracker()
    if 'camera_active' not in st.session_state:
        st.session_state.camera_active = False
    
    # Sidebar controls
    st.sidebar.header("🎛️ Camera Controls")
    
    # Camera selection
    camera_index = st.sidebar.selectbox("Select Camera:", [0, 1, 2], index=0)
    
    # Camera control buttons
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("📹 Start Camera"):
            if st.session_state.tracker.start_camera(camera_index):
                st.session_state.camera_active = True
                st.sidebar.success("Camera started!")
            else:
                st.sidebar.error("Failed to start camera!")
    
    with col2:
        if st.button("⏹️ Stop Camera"):
            st.session_state.tracker.stop_camera()
            st.session_state.camera_active = False
            st.sidebar.info("Camera stopped!")
    
    # Detection settings
    st.sidebar.subheader("⚙️ Detection Settings")
    
    min_area = st.sidebar.slider("Minimum Object Size:", 100, 2000, 300, 50)
    st.session_state.min_area = min_area
    
    kernel_size = st.sidebar.slider("Noise Filter:", 3, 15, 5, 2)
    st.session_state.kernel_size = kernel_size
    
    trail_thickness = st.sidebar.slider("Trail Thickness:", 1, 5, 2)
    st.session_state.trail_thickness = trail_thickness
    
    if st.sidebar.button("🧹 Clear All Trails"):
        st.session_state.tracker.tracked_objects.clear()
    
    # Main content area
    if st.session_state.camera_active:
        # Video display
        st.subheader("📹 Live Multi-Color Tracking")
        video_placeholder = st.empty()
        
        # Stats area
        stats_col1, stats_col2, stats_col3 = st.columns(3)
        
        with stats_col1:
            status_placeholder = st.empty()
        with stats_col2:
            objects_placeholder = st.empty()
        with stats_col3:
            colors_placeholder = st.empty()
        
        # Real-time processing loop
        try:
            status_placeholder.success("🟢 Camera Active - Detecting All Colors")
            
            while st.session_state.camera_active:
                # Get latest frame from camera
                frame = st.session_state.tracker.get_latest_frame()
                
                if frame is not None:
                    # Process frame for multi-color detection
                    tracked_frame, detected_objects = st.session_state.tracker.process_frame(frame)
                    
                    if tracked_frame is not None:
                        # Convert for display
                        tracked_rgb = cv2.cvtColor(tracked_frame, cv2.COLOR_BGR2RGB)
                        video_placeholder.image(tracked_rgb, channels="RGB", use_column_width=True)
                        
                        # Update stats
                        objects_placeholder.metric("🎯 Objects Detected", len(detected_objects))
                        
                        if detected_objects:
                            colors_list = [obj['color'] for obj in detected_objects]
                            unique_colors = len(set(colors_list))
                            colors_placeholder.metric("🌈 Unique Colors", unique_colors)
                        else:
                            colors_placeholder.metric("🌈 Unique Colors", 0)
                
                # Small delay to prevent overwhelming the interface
                time.sleep(0.05)
                
        except Exception as e:
            st.error(f"Error during tracking: {str(e)}")
            st.session_state.camera_active = False
            st.session_state.tracker.stop_camera()
            
    else:
        st.info("👆 Click 'Start Camera' in the sidebar to begin multi-color tracking")
        
        # Color legend
        st.subheader("🎨 Detectable Colors")
        
        color_cols = st.columns(4)
        colors = ['Red', 'Orange', 'Yellow', 'Green', 'Cyan', 'Blue', 'Purple', 'Pink']
        
        for i, color in enumerate(colors):
            with color_cols[i % 4]:
                st.markdown(f"**{color}**")

if __name__ == "__main__":
    main()