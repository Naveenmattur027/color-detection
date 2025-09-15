import streamlit as st
import cv2
import numpy as np
from PIL import Image

class MajorColorDetector:
    def __init__(self):
        """Initialize with comprehensive color database - focus on major colors only"""
        
        # Comprehensive HSV color ranges for maximum detection
        self.color_ranges = {
            # REDS
            'red': [(np.array([0, 120, 70]), np.array([10, 255, 255])), 
                    (np.array([170, 120, 70]), np.array([180, 255, 255]))],
            'crimson': [(np.array([0, 150, 100]), np.array([15, 255, 255]))],
            'scarlet': [(np.array([0, 200, 150]), np.array([10, 255, 255]))],
            'cherry': [(np.array([170, 100, 80]), np.array([180, 255, 255]))],
            'rose': [(np.array([0, 80, 150]), np.array([15, 200, 255]))],
            'brick': [(np.array([0, 100, 50]), np.array([15, 255, 180]))],
            'burgundy': [(np.array([0, 150, 30]), np.array([15, 255, 120]))],
            'maroon': [(np.array([0, 100, 30]), np.array([15, 255, 100]))],
            'ruby': [(np.array([170, 180, 100]), np.array([180, 255, 255]))],
            'coral': [(np.array([0, 100, 150]), np.array([20, 255, 255]))],
            
            # ORANGES
            'orange': [(np.array([10, 100, 100]), np.array([25, 255, 255]))],
            'tangerine': [(np.array([12, 150, 150]), np.array([22, 255, 255]))],
            'peach': [(np.array([8, 80, 180]), np.array([25, 200, 255]))],
            'apricot': [(np.array([15, 100, 200]), np.array([25, 180, 255]))],
            'amber': [(np.array([20, 100, 150]), np.array([30, 255, 255]))],
            'copper': [(np.array([15, 120, 80]), np.array([25, 255, 200]))],
            'bronze': [(np.array([18, 100, 60]), np.array([30, 255, 150]))],
            'rust': [(np.array([10, 150, 80]), np.array([20, 255, 180]))],
            'papaya': [(np.array([18, 120, 200]), np.array([28, 200, 255]))],
            'mango': [(np.array([20, 150, 180]), np.array([30, 255, 255]))],
            
            # YELLOWS
            'yellow': [(np.array([20, 100, 100]), np.array([35, 255, 255]))],
            'gold': [(np.array([22, 120, 150]), np.array([35, 255, 255]))],
            'lemon': [(np.array([25, 150, 180]), np.array([35, 255, 255]))],
            'canary': [(np.array([25, 180, 200]), np.array([35, 255, 255]))],
            'banana': [(np.array([22, 100, 180]), np.array([35, 200, 255]))],
            'cream': [(np.array([20, 30, 200]), np.array([35, 100, 255]))],
            'butter': [(np.array([25, 80, 200]), np.array([35, 150, 255]))],
            'ivory': [(np.array([20, 20, 220]), np.array([35, 80, 255]))],
            'champagne': [(np.array([25, 40, 180]), np.array([35, 120, 255]))],
            'mustard': [(np.array([25, 120, 100]), np.array([35, 255, 200]))],
            
            # GREENS
            'green': [(np.array([40, 50, 50]), np.array([80, 255, 255]))],
            'lime': [(np.array([60, 120, 120]), np.array([80, 255, 255]))],
            'forest': [(np.array([50, 100, 30]), np.array([70, 255, 120]))],
            'emerald': [(np.array([55, 150, 80]), np.array([75, 255, 200]))],
            'jade': [(np.array([60, 80, 100]), np.array([80, 200, 200]))],
            'mint': [(np.array([75, 50, 180]), np.array([85, 150, 255]))],
            'olive': [(np.array([35, 100, 50]), np.array([55, 255, 150]))],
            'sage': [(np.array([70, 30, 120]), np.array([90, 100, 200]))],
            'pine': [(np.array([45, 120, 40]), np.array([65, 255, 120]))],
            'moss': [(np.array([50, 80, 60]), np.array([70, 200, 140]))],
            'kelly': [(np.array([50, 180, 100]), np.array([70, 255, 200]))],
            'chartreuse': [(np.array([65, 100, 150]), np.array([85, 255, 255]))],
            
            # CYANS/TEALS
            'cyan': [(np.array([80, 100, 100]), np.array([100, 255, 255]))],
            'teal': [(np.array([85, 120, 80]), np.array([95, 255, 200]))],
            'turquoise': [(np.array([85, 100, 150]), np.array([95, 255, 255]))],
            'aqua': [(np.array([82, 80, 180]), np.array([98, 200, 255]))],
            'seafoam': [(np.array([80, 60, 160]), np.array([100, 150, 255]))],
            'mint_blue': [(np.array([85, 40, 200]), np.array([95, 120, 255]))],
            'powder_blue': [(np.array([90, 30, 220]), np.array([110, 80, 255]))],
            
            # BLUES
            'blue': [(np.array([100, 50, 50]), np.array([130, 255, 255]))],
            'navy': [(np.array([110, 100, 30]), np.array([130, 255, 100]))],
            'royal': [(np.array([115, 150, 100]), np.array([125, 255, 200]))],
            'sky': [(np.array([105, 80, 180]), np.array([125, 200, 255]))],
            'azure': [(np.array([110, 100, 200]), np.array([125, 180, 255]))],
            'cobalt': [(np.array([115, 180, 80]), np.array([125, 255, 180]))],
            'sapphire': [(np.array([115, 150, 60]), np.array([125, 255, 150]))],
            'steel': [(np.array([105, 80, 100]), np.array([125, 150, 200]))],
            'denim': [(np.array([110, 120, 60]), np.array([125, 200, 140]))],
            'periwinkle': [(np.array([120, 60, 180]), np.array([135, 150, 255]))],
            'cornflower': [(np.array([115, 100, 180]), np.array([130, 200, 255]))],
            
            # PURPLES/VIOLETS
            'purple': [(np.array([130, 50, 50]), np.array([170, 255, 255]))],
            'violet': [(np.array([125, 100, 100]), np.array([145, 255, 255]))],
            'indigo': [(np.array([105, 100, 50]), np.array([125, 255, 150]))],
            'plum': [(np.array([140, 80, 80]), np.array([160, 200, 200]))],
            'lavender': [(np.array([135, 40, 180]), np.array([155, 120, 255]))],
            'orchid': [(np.array([140, 100, 150]), np.array([160, 255, 255]))],
            'amethyst': [(np.array([135, 120, 100]), np.array([155, 255, 200]))],
            'lilac': [(np.array([140, 50, 200]), np.array([160, 150, 255]))],
            'mauve': [(np.array([145, 80, 120]), np.array([165, 180, 220]))],
            'eggplant': [(np.array([140, 150, 40]), np.array([160, 255, 120]))],
            'grape': [(np.array([135, 100, 60]), np.array([155, 255, 150]))],
            
            # PINKS/MAGENTAS
            'pink': [(np.array([140, 50, 150]), np.array([170, 255, 255]))],
            'magenta': [(np.array([145, 100, 100]), np.array([165, 255, 255]))],
            'fuchsia': [(np.array([150, 150, 150]), np.array([170, 255, 255]))],
            'hot_pink': [(np.array([155, 180, 180]), np.array([175, 255, 255]))],
            'rose_pink': [(np.array([0, 80, 180]), np.array([15, 200, 255]))],
            'blush': [(np.array([0, 40, 220]), np.array([15, 120, 255]))],
            'salmon': [(np.array([0, 100, 180]), np.array([20, 200, 255]))],
            'flamingo': [(np.array([155, 120, 200]), np.array([175, 255, 255]))],
            'bubblegum': [(np.array([160, 100, 220]), np.array([175, 200, 255]))],
            'carnation': [(np.array([0, 60, 200]), np.array([15, 150, 255]))],
            
            # BROWNS/TANS
            'brown': [(np.array([10, 50, 20]), np.array([25, 255, 200]))],
            'tan': [(np.array([15, 80, 120]), np.array([30, 180, 220]))],
            'beige': [(np.array([15, 30, 180]), np.array([30, 100, 255]))],
            'khaki': [(np.array([25, 60, 140]), np.array([35, 150, 220]))],
            'chocolate': [(np.array([12, 120, 50]), np.array([22, 255, 150]))],
            'coffee': [(np.array([15, 100, 40]), np.array([25, 255, 120]))],
            'caramel': [(np.array([18, 100, 100]), np.array([28, 200, 200]))],
            'cinnamon': [(np.array([15, 120, 80]), np.array([25, 255, 180]))],
            'chestnut': [(np.array([12, 150, 60]), np.array([22, 255, 140]))],
            'mahogany': [(np.array([8, 120, 40]), np.array([18, 255, 100]))],
            'sienna': [(np.array([15, 150, 70]), np.array([25, 255, 160]))],
            
            # NEUTRALS
            'white': [(np.array([0, 0, 200]), np.array([180, 30, 255]))],
            'black': [(np.array([0, 0, 0]), np.array([180, 255, 50]))],
            'gray': [(np.array([0, 0, 50]), np.array([180, 30, 200]))],
            'silver': [(np.array([0, 0, 120]), np.array([180, 20, 220]))],
            'charcoal': [(np.array([0, 0, 30]), np.array([180, 50, 100]))],
            'slate': [(np.array([200, 10, 80]), np.array([220, 50, 150]))],
            'pearl': [(np.array([0, 0, 230]), np.array([180, 20, 255]))],
            'ash': [(np.array([0, 0, 100]), np.array([180, 40, 180]))],
            'smoke': [(np.array([0, 0, 120]), np.array([180, 30, 180]))],
            
            # SPECIALTY COLORS
            'neon_green': [(np.array([60, 200, 200]), np.array([80, 255, 255]))],
            'neon_pink': [(np.array([160, 200, 200]), np.array([175, 255, 255]))],
            'neon_blue': [(np.array([110, 200, 200]), np.array([125, 255, 255]))],
            'neon_yellow': [(np.array([25, 200, 200]), np.array([35, 255, 255]))],
            'electric_blue': [(np.array([115, 255, 255]), np.array([125, 255, 255]))],
            'hot_magenta': [(np.array([155, 255, 200]), np.array([170, 255, 255]))],
            'lime_green': [(np.array([70, 200, 180]), np.array([85, 255, 255]))],
        }
        
        # INCREASED thresholds for major content detection only
        self.min_area = 2000  # Increased from 300 to 2000
        self.min_width = 50   # Minimum width for valid detection
        self.min_height = 50  # Minimum height for valid detection
        self.area_threshold_percent = 0.5  # Must be at least 0.5% of image area
    
    def preprocess_image(self, image):
        """Preprocess image for better color detection"""
        if isinstance(image, Image.Image):
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Apply bilateral filter for noise reduction
        filtered = cv2.bilateralFilter(image, 9, 75, 75)
        
        # Enhance contrast
        lab = cv2.cvtColor(filtered, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # Convert to HSV
        hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
        
        return image, hsv
    
    def create_color_mask(self, hsv_image, color_name):
        """Create mask for specific color with aggressive filtering"""
        masks = []
        
        for lower, upper in self.color_ranges[color_name]:
            mask = cv2.inRange(hsv_image, lower, upper)
            masks.append(mask)
        
        # Combine masks
        combined_mask = masks[0]
        for mask in masks[1:]:
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        # ENHANCED morphological operations to remove small particles
        kernel_small = np.ones((3, 3), np.uint8)
        kernel_large = np.ones((7, 7), np.uint8)  # Larger kernel for more aggressive filtering
        
        # Multiple opening operations to remove small noise
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_small, iterations=2)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_large, iterations=1)
        
        # Closing to fill gaps in major objects
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_large, iterations=2)
        
        # Median blur to smooth edges
        combined_mask = cv2.medianBlur(combined_mask, 9)  # Increased blur radius
        
        return combined_mask
    
    def filter_major_contours(self, contours, image_area):
        """Filter contours to keep only major content"""
        major_contours = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate area percentage of total image
            area_percent = (area / image_area) * 100
            
            # Multiple filtering criteria
            is_large_enough = area > self.min_area
            is_wide_enough = w > self.min_width
            is_tall_enough = h > self.min_height
            is_significant_area = area_percent > self.area_threshold_percent
            
            # Aspect ratio check - avoid very thin lines or weird shapes
            aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else float('inf')
            is_reasonable_shape = aspect_ratio < 10  # Not too elongated
            
            # Solidity check - how "solid" the shape is (removes scattered pixels)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            is_solid = solidity > 0.3  # At least 30% solid
            
            if (is_large_enough and is_wide_enough and is_tall_enough and 
                is_significant_area and is_reasonable_shape and is_solid):
                major_contours.append(contour)
        
        return major_contours
    
    def merge_nearby_detections(self, detected_colors, merge_distance=100):
        """Merge nearby detections of the same color to reduce redundancy"""
        if not detected_colors:
            return detected_colors
        
        merged_colors = []
        processed = set()
        
        for i, detection in enumerate(detected_colors):
            if i in processed:
                continue
                
            current_color = detection['color']
            current_pos = detection['position']
            merged_area = detection['area']
            merged_detections = [detection]
            
            # Find nearby detections of same color
            for j, other_detection in enumerate(detected_colors[i+1:], i+1):
                if j in processed:
                    continue
                    
                if other_detection['color'] == current_color:
                    other_pos = other_detection['position']
                    distance = np.sqrt((current_pos[0] - other_pos[0])**2 + 
                                     (current_pos[1] - other_pos[1])**2)
                    
                    if distance < merge_distance:
                        merged_area += other_detection['area']
                        merged_detections.append(other_detection)
                        processed.add(j)
            
            # Create merged detection
            if len(merged_detections) > 1:
                # Calculate average position weighted by area
                total_area = sum(d['area'] for d in merged_detections)
                avg_x = sum(d['position'][0] * d['area'] for d in merged_detections) / total_area
                avg_y = sum(d['position'][1] * d['area'] for d in merged_detections) / total_area
                
                merged_detection = {
                    'color': current_color,
                    'area': merged_area,
                    'confidence': min(100, int((merged_area / 2000) * 20)),
                    'position': (int(avg_x), int(avg_y)),
                    'dimensions': merged_detections[0]['dimensions'],  # Use first one's dimensions
                    'merged_count': len(merged_detections)
                }
            else:
                merged_detection = detection
                merged_detection['merged_count'] = 1
            
            merged_colors.append(merged_detection)
            processed.add(i)
        
        return merged_colors
    
    def detect_colors(self, image):
        """Detect only major colors in the image"""
        original_image, hsv_image = self.preprocess_image(image)
        detected_colors = []
        result_image = original_image.copy()
        
        # Calculate image area for percentage calculations
        image_area = original_image.shape[0] * original_image.shape[1]
        
        # Color mapping for visualization (BGR)
        color_bgr_map = {
            # Reds
            'red': (0, 0, 255), 'crimson': (20, 20, 220), 'scarlet': (0, 20, 255),
            'cherry': (0, 30, 200), 'rose': (80, 80, 255), 'brick': (40, 40, 180),
            'burgundy': (20, 0, 100), 'maroon': (0, 0, 128), 'ruby': (0, 40, 200),
            'coral': (80, 127, 255),
            
            # Oranges
            'orange': (0, 165, 255), 'tangerine': (0, 140, 255), 'peach': (180, 200, 255),
            'apricot': (180, 215, 255), 'amber': (0, 191, 255), 'copper': (72, 118, 184),
            'bronze': (40, 120, 205), 'rust': (41, 69, 183), 'papaya': (113, 179, 255),
            'mango': (96, 190, 255),
            
            # Yellows
            'yellow': (0, 255, 255), 'gold': (0, 215, 255), 'lemon': (0, 250, 255),
            'canary': (0, 255, 255), 'banana': (140, 255, 255), 'cream': (220, 255, 245),
            'butter': (180, 240, 255), 'ivory': (240, 255, 255), 'champagne': (207, 248, 247),
            'mustard': (0, 219, 255),
            
            # Greens
            'green': (0, 255, 0), 'lime': (0, 255, 128), 'forest': (34, 139, 34),
            'emerald': (80, 200, 120), 'jade': (100, 180, 150), 'mint': (152, 255, 152),
            'olive': (0, 128, 128), 'sage': (158, 189, 147), 'pine': (1, 121, 111),
            'moss': (100, 148, 125), 'kelly': (76, 187, 23), 'chartreuse': (0, 255, 127),
            
            # Cyans/Teals
            'cyan': (255, 255, 0), 'teal': (128, 128, 0), 'turquoise': (208, 224, 64),
            'aqua': (255, 255, 0), 'seafoam': (159, 226, 191), 'mint_blue': (175, 238, 238),
            'powder_blue': (230, 224, 176),
            
            # Blues
            'blue': (255, 0, 0), 'navy': (128, 0, 0), 'royal': (225, 105, 65),
            'sky': (235, 206, 135), 'azure': (255, 255, 240), 'cobalt': (153, 76, 0),
            'sapphire': (146, 82, 15), 'steel': (180, 130, 70), 'denim': (151, 96, 21),
            'periwinkle': (197, 203, 199), 'cornflower': (237, 149, 100),
            
            # Purples/Violets
            'purple': (128, 0, 128), 'violet': (238, 130, 238), 'indigo': (130, 0, 75),
            'plum': (173, 100, 221), 'lavender': (250, 230, 230), 'orchid': (214, 112, 218),
            'amethyst': (204, 153, 153), 'lilac': (200, 162, 200), 'mauve': (176, 196, 222),
            'eggplant': (97, 64, 81), 'grape': (111, 45, 168),
            
            # Pinks/Magentas
            'pink': (203, 192, 255), 'magenta': (255, 0, 255), 'fuchsia': (255, 0, 255),
            'hot_pink': (180, 105, 255), 'rose_pink': (255, 102, 204), 'blush': (254, 192, 203),
            'salmon': (114, 128, 250), 'flamingo': (252, 142, 172), 'bubblegum': (255, 193, 204),
            'carnation': (255, 166, 201),
            
            # Browns/Tans
            'brown': (42, 42, 165), 'tan': (140, 180, 210), 'beige': (245, 245, 220),
            'khaki': (189, 183, 107), 'chocolate': (30, 105, 210), 'coffee': (111, 78, 55),
            'caramel': (175, 200, 255), 'cinnamon': (123, 63, 0), 'chestnut': (149, 69, 53),
            'mahogany': (192, 64, 0), 'sienna': (45, 82, 160),
            
            # Neutrals
            'white': (255, 255, 255), 'black': (50, 50, 50), 'gray': (128, 128, 128),
            'silver': (192, 192, 192), 'charcoal': (54, 69, 79), 'slate': (112, 128, 144),
            'pearl': (234, 224, 200), 'ash': (178, 190, 181), 'smoke': (115, 130, 118),
            
            # Specialty Colors
            'neon_green': (0, 255, 0), 'neon_pink': (255, 20, 147), 'neon_blue': (30, 144, 255),
            'neon_yellow': (255, 255, 0), 'electric_blue': (125, 249, 255), 'hot_magenta': (255, 29, 206),
            'lime_green': (50, 205, 50),
        }
        
        # Process each color
        for color_name in self.color_ranges.keys():
            mask = self.create_color_mask(hsv_image, color_name)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter to get only major contours
            major_contours = self.filter_major_contours(contours, image_area)
            
            color_bgr = color_bgr_map.get(color_name, (255, 255, 255))
            
            for contour in major_contours:
                area = cv2.contourArea(contour)
                x, y, w, h = cv2.boundingRect(contour)
                center_x = x + w // 2
                center_y = y + h // 2
                
                # Calculate confidence based on area and image percentage
                area_percent = (area / image_area) * 100
                confidence = min(100, int(area_percent * 10))  # Better confidence calculation
                
                # Draw detection with thicker lines for major objects
                cv2.rectangle(result_image, (x, y), (x + w, y + h), color_bgr, 4)
                cv2.circle(result_image, (center_x, center_y), 8, color_bgr, -1)
                
                # Add label with larger font for major detections
                label = color_name.replace('_', ' ').upper()
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                cv2.rectangle(result_image, (x, y - label_size[1] - 15), 
                            (x + label_size[0] + 10, y), color_bgr, -1)
                cv2.putText(result_image, label, (x + 5, y - 8), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # Add area percentage info
                area_text = f"{area_percent:.1f}%"
                cv2.putText(result_image, area_text, (x, y + h + 20), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2)
                
                detected_colors.append({
                    'color': color_name.replace('_', ' ').title(),
                    'area': int(area),
                    'area_percent': round(area_percent, 2),
                    'confidence': confidence,
                    'position': (center_x, center_y),
                    'dimensions': (w, h)
                })
        
        # Merge nearby detections to reduce redundancy
        detected_colors = self.merge_nearby_detections(detected_colors)
        
        return result_image, detected_colors

def main():
    st.set_page_config(page_title="Major Color Detection - Focus on Main Content", layout="wide")
    
    # Initialize detector
    if 'detector' not in st.session_state:
        st.session_state.detector = MajorColorDetector()
    
    st.title("🎯 Major Color Detection System")
    st.markdown("**Advanced AI-powered detection focusing only on major color content**")
    st.info("🔍 This system filters out small particles and focuses only on major color areas (>0.5% of image)")
    
    # Settings sidebar
    st.sidebar.header("🛠️ Detection Settings")
    
    # Allow user to adjust thresholds
    min_area = st.sidebar.slider("Minimum Area (pixels)", 500, 5000, 2000)
    area_threshold = st.sidebar.slider("Area Threshold (%)", 0.1, 2.0, 0.5, 0.1)
    merge_distance = st.sidebar.slider("Merge Distance (pixels)", 50, 200, 100)
    
    # Update detector settings
    st.session_state.detector.min_area = min_area
    st.session_state.detector.area_threshold_percent = area_threshold
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**📋 Filtering Criteria:**")
    st.sidebar.markdown(f"• Minimum area: {min_area:,} pixels")
    st.sidebar.markdown(f"• Must be >{area_threshold}% of image")
    st.sidebar.markdown("• Minimum 50x50 pixels")
    st.sidebar.markdown("• Aspect ratio < 10:1")
    st.sidebar.markdown("• Solidity > 30%")
    
    # Camera input
    uploaded_file = st.camera_input("📸 Capture Image for Major Color Detection")
    
    # Start detection button
    if st.button("🔍 Detect Major Colors Only", type="primary", use_container_width=True):
        if uploaded_file is not None:
            try:
                # Process the image
                image = Image.open(uploaded_file)
                
                with st.spinner("🔍 Analyzing major color content..."):
                    result_image, detected_colors = st.session_state.detector.detect_colors(image)
                
                # Convert result back to RGB for display
                result_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📷 Original Image")
                    st.image(image, use_container_width=True)
                
                with col2:
                    st.subheader("🎯 Major Colors Detected")
                    st.image(result_rgb, use_container_width=True)
                
                # Show results
                if detected_colors:
                    st.subheader("📊 Major Color Detection Results")
                    st.success(f"🎯 **{len(detected_colors)} major color regions detected!**")
                    
                    # Sort by area percentage (largest first)
                    detected_colors.sort(key=lambda x: x['area_percent'], reverse=True)
                    
                    # Display detailed results
                    st.subheader("🏆 Top Major Color Regions")
                    
                    for i, detection in enumerate(detected_colors[:6], 1):  # Show top 6
                        with st.container():
                            cols = st.columns([1, 3, 2, 2, 2])
                            
                            with cols[0]:
                                st.markdown(f"**#{i}**")
                            
                            with cols[1]:
                                st.markdown(f"**🎨 {detection['color']}**")
                                if 'merged_count' in detection and detection['merged_count'] > 1:
                                    st.caption(f"(Merged from {detection['merged_count']} regions)")
                            
                            with cols[2]:
                                st.metric("Area", f"{detection['area']:,} px")
                                st.caption(f"{detection['area_percent']}% of image")
                            
                            with cols[3]:
                                st.metric("Confidence", f"{detection['confidence']}%")
                                st.progress(detection['confidence']/100)
                            
                            with cols[4]:
                                w, h = detection['dimensions']
                                st.metric("Size", f"{w}×{h}")
                                st.caption(f"Position: {detection['position']}")
                            
                            st.divider()
                    
                    if len(detected_colors) > 6:
                        st.info(f"📋 Showing top 6 major regions. Total detected: {len(detected_colors)}")
                    
                    # Enhanced Statistics
                    st.subheader("📈 Detailed Analysis")
                    
                    # Main stats row
                    stat_cols = st.columns(4)
                    with stat_cols[0]:
                        st.metric("Major Regions", len(detected_colors))
                    with stat_cols[1]:
                        unique_colors = len(set(d['color'] for d in detected_colors))
                        st.metric("Unique Colors", unique_colors)
                    with stat_cols[2]:
                        total_coverage = sum(d['area_percent'] for d in detected_colors)
                        st.metric("Total Coverage", f"{total_coverage:.1f}%")
                    with stat_cols[3]:
                        avg_size = sum(d['area'] for d in detected_colors) / len(detected_colors)
                        st.metric("Avg Region Size", f"{avg_size:,.0f} px")
                    
                    # Coverage breakdown
                    st.subheader("📊 Color Coverage Analysis")
                    
                    coverage_data = {}
                    for detection in detected_colors:
                        color = detection['color']
                        if color in coverage_data:
                            coverage_data[color] += detection['area_percent']
                        else:
                            coverage_data[color] = detection['area_percent']
                    
                    # Sort by coverage
                    sorted_coverage = sorted(coverage_data.items(), key=lambda x: x[1], reverse=True)
                    
                    coverage_cols = st.columns(2)
                    with coverage_cols[0]:
                        st.markdown("**🏆 Top Colors by Coverage:**")
                        for color, coverage in sorted_coverage[:5]:
                            st.markdown(f"• **{color}**: {coverage:.2f}%")
                            st.progress(coverage/20)  # Scale for visualization
                    
                    with coverage_cols[1]:
                        st.markdown("**📏 Size Distribution:**")
                        large_regions = len([d for d in detected_colors if d['area_percent'] > 2])
                        medium_regions = len([d for d in detected_colors if 1 <= d['area_percent'] <= 2])
                        small_regions = len([d for d in detected_colors if d['area_percent'] < 1])
                        
                        st.markdown(f"• **Large regions (>2%)**: {large_regions}")
                        st.markdown(f"• **Medium regions (1-2%)**: {medium_regions}")
                        st.markdown(f"• **Small regions (<1%)**: {small_regions}")
                    
                    # Quality indicators
                    st.subheader("✅ Detection Quality")
                    quality_cols = st.columns(3)
                    
                    with quality_cols[0]:
                        high_conf = len([d for d in detected_colors if d['confidence'] > 70])
                        st.metric("High Confidence", f"{high_conf}/{len(detected_colors)}")
                        st.caption("(>70% confidence)")
                    
                    with quality_cols[1]:
                        large_areas = len([d for d in detected_colors if d['area'] > 5000])
                        st.metric("Large Areas", f"{large_areas}/{len(detected_colors)}")
                        st.caption("(>5000 pixels)")
                    
                    with quality_cols[2]:
                        merged_count = sum(d.get('merged_count', 1) for d in detected_colors)
                        st.metric("Total Regions", merged_count)
                        st.caption("(before merging)")
                        
                else:
                    st.warning("🔍 No major color regions detected!")
                    st.markdown("**Possible reasons:**")
                    st.markdown("• Image may contain only small color particles")
                    st.markdown("• Colors may be too scattered or fragmented") 
                    st.markdown("• Try adjusting the detection thresholds in the sidebar")
                    st.markdown("• Ensure good lighting and clear color boundaries")
                    
                    st.info("💡 **Tips for better detection:**")
                    st.markdown("• Lower the minimum area threshold")
                    st.markdown("• Use images with distinct, solid color objects")
                    st.markdown("• Ensure objects are at least 50x50 pixels")
                    st.markdown("• Avoid images with too much texture or patterns")
                    
            except Exception as e:
                st.error(f"❌ Error processing image: {str(e)}")
                st.exception(e)  # Show full error for debugging
        else:
            st.error("📸 Please capture a photo first!")
            
    # Additional info section
    with st.expander("ℹ️ How Major Color Detection Works"):
        st.markdown("""
        **🎯 Advanced Filtering System:**
        
        **1. Size Filtering:**
        - Minimum area threshold (default: 2000 pixels)
        - Minimum dimensions (50x50 pixels)
        - Must occupy significant portion of image (>0.5%)
        
        **2. Shape Analysis:**
        - Aspect ratio check (width:height < 10:1)
        - Solidity measurement (>30% solid)
        - Removes scattered pixels and thin lines
        
        **3. Morphological Processing:**
        - Multiple opening operations remove noise
        - Closing operations fill gaps in major objects
        - Enhanced median filtering for smooth edges
        
        **4. Smart Merging:**
        - Combines nearby regions of same color
        - Reduces redundant detections
        - Provides cleaner final results
        
        **5. Quality Metrics:**
        - Confidence based on area percentage
        - Coverage analysis shows dominant colors
        - Detailed statistics for each detection
        """)

if __name__ == "__main__":
    main()
