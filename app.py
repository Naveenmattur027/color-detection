import streamlit as st
import cv2
import numpy as np
from PIL import Image

class UltraColorDetector:
    def __init__(self):
        """Initialize with comprehensive color database - 100+ colors"""
        
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
        
        # Detection settings
        self.min_area = 300
    
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
        """Create mask for specific color"""
        masks = []
        
        for lower, upper in self.color_ranges[color_name]:
            mask = cv2.inRange(hsv_image, lower, upper)
            masks.append(mask)
        
        # Combine masks
        combined_mask = masks[0]
        for mask in masks[1:]:
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        # Apply morphological operations
        kernel = np.ones((3, 3), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.medianBlur(combined_mask, 5)
        
        return combined_mask
    
    def detect_colors(self, image):
        """Detect all colors in the image"""
        original_image, hsv_image = self.preprocess_image(image)
        detected_colors = []
        result_image = original_image.copy()
        
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
            
            color_bgr = color_bgr_map.get(color_name, (255, 255, 255))
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                if area > self.min_area:
                    # Get bounding rectangle
                    x, y, w, h = cv2.boundingRect(contour)
                    center_x = x + w // 2
                    center_y = y + h // 2
                    
                    # Calculate confidence
                    confidence = min(100, int((area / 1000) * 20))
                    
                    # Draw detection
                    cv2.rectangle(result_image, (x, y), (x + w, y + h), color_bgr, 3)
                    cv2.circle(result_image, (center_x, center_y), 5, color_bgr, -1)
                    
                    # Add label
                    label = color_name.replace('_', ' ').upper()
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(result_image, (x, y - label_size[1] - 10), 
                                (x + label_size[0], y), color_bgr, -1)
                    cv2.putText(result_image, label, (x, y - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    detected_colors.append({
                        'color': color_name.replace('_', ' ').title(),
                        'area': int(area),
                        'confidence': confidence,
                        'position': (center_x, center_y),
                        'dimensions': (w, h)
                    })
        
        return result_image, detected_colors

def main():
    st.set_page_config(page_title="Ultra Color Detection - 70+ Colors", layout="wide")
    
    # Initialize detector
    if 'detector' not in st.session_state:
        st.session_state.detector = UltraColorDetector()
    
    st.title("🌈 Ultra Color Detection - 70+ Colors")
    st.markdown("**Advanced AI-powered color detection system with 70+ trained colors**")
    
    # Camera input
    uploaded_file = st.camera_input("📸 Capture Image")
    
    # Start detection button
    if st.button("🔍 Start Ultra Detection", type="primary", use_container_width=True):
        if uploaded_file is not None:
            try:
                # Process the image
                image = Image.open(uploaded_file)
                
                with st.spinner("🤖 Analyzing image with 70+ color models..."):
                    result_image, detected_colors = st.session_state.detector.detect_colors(image)
                
                # Convert result back to RGB for display
                result_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📷 Original Image")
                    st.image(image, use_container_width=True)
                
                with col2:
                    st.subheader("🎨 Detected Colors")
                    st.image(result_rgb, use_container_width=True)
                
                # Show results
                if detected_colors:
                    st.subheader("📊 Detection Results")
                    st.success(f"🎯 **{len(detected_colors)} color objects detected!**")
                    
                    # Sort by confidence
                    detected_colors.sort(key=lambda x: x['confidence'], reverse=True)
                    
                    # Display top detections
                    cols = st.columns(3)
                    for i, detection in enumerate(detected_colors[:9]):  # Show top 9
                        with cols[i % 3]:
                            st.markdown(f"**🎨 {detection['color']}**")
                            st.markdown(f"📏 Area: {detection['area']:,} px")
                            st.markdown(f"🎯 Confidence: {detection['confidence']}%")
                            st.progress(detection['confidence']/100)
                            st.markdown("---")
                    
                    if len(detected_colors) > 9:
                        st.info(f"Showing top 9 detections. Total: {len(detected_colors)}")
                    
                    # Statistics
                    st.subheader("📈 Statistics")
                    stat_cols = st.columns(4)
                    with stat_cols[0]:
                        st.metric("Total Objects", len(detected_colors))
                    with stat_cols[1]:
                        unique_colors = len(set(d['color'] for d in detected_colors))
                        st.metric("Unique Colors", unique_colors)
                    with stat_cols[2]:
                        avg_conf = sum(d['confidence'] for d in detected_colors) / len(detected_colors)
                        st.metric("Avg Confidence", f"{avg_conf:.1f}%")
                    with stat_cols[3]:
                        total_area = sum(d['area'] for d in detected_colors)
                        st.metric("Total Area", f"{total_area:,}")
                        
                else:
                    st.warning("🔍 No colors detected. Try:")
                    st.markdown("• Better lighting")
                    st.markdown("• Closer objects") 
                    st.markdown("• Solid color objects")
                    st.markdown("• Clean background")
                    
            except Exception as e:
                st.error(f"❌ Error processing image: {str(e)}")
        else:
            st.error("📸 Please capture a photo first!")
    
    # Color information sidebar
    with st.sidebar:
        st.header("🌈 Supported Colors (70+)")
        
        color_categories = {
            "🔴 Reds": ["Red", "Crimson", "Scarlet", "Cherry", "Rose", "Burgundy", "Maroon", "Ruby", "Coral", "Brick"],
            "🟠 Oranges": ["Orange", "Tangerine", "Peach", "Apricot", "Amber", "Copper", "Bronze", "Rust", "Papaya", "Mango"],
            "🟡 Yellows": ["Yellow", "Gold", "Lemon", "Canary", "Banana", "Cream", "Butter", "Ivory", "Champagne", "Mustard"],
            "🟢 Greens": ["Green", "Lime", "Forest", "Emerald", "Jade", "Mint", "Olive", "Sage", "Pine", "Moss", "Kelly", "Chartreuse"],
            "🔵 Blues": ["Blue", "Navy", "Royal", "Sky", "Azure", "Cobalt", "Sapphire", "Steel", "Denim", "Periwinkle", "Cornflower"],
            "🟣 Purples": ["Purple", "Violet", "Indigo", "Plum", "Lavender", "Orchid", "Amethyst", "Lilac", "Mauve", "Eggplant", "Grape"],
            "🩷 Pinks": ["Pink", "Magenta", "Fuchsia", "Hot Pink", "Rose Pink", "Blush", "Salmon", "Flamingo", "Bubblegum", "Carnation"],
            "🤎 Browns": ["Brown", "Tan", "Beige", "Khaki", "Chocolate", "Coffee", "Caramel", "Cinnamon", "Chestnut", "Mahogany", "Sienna"],
            "⚫ Neutrals": ["White", "Black", "Gray", "Silver", "Charcoal", "Slate", "Pearl", "Ash", "Smoke"],
            "✨ Special": ["Neon Green", "Neon Pink", "Neon Blue", "Neon Yellow", "Electric Blue", "Hot Magenta", "Lime Green"]
        }
        
        for category, colors in color_categories.items():
            with st.expander(category):
                for color in colors:
                    st.markdown(f"• {color}")
        
        st.markdown("---")
        st.markdown("**🎯 Features:**")
        st.markdown("• 70+ Color Detection")
        st.markdown("• HSV Color Analysis")
        st.markdown("• Noise Reduction")
        st.markdown("• Confidence Scoring")
        st.markdown("• Real-time Processing")

if __name__ == "__main__":
    main()
