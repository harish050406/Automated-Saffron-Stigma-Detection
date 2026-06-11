import cv2
import numpy as np

class StigmaAnalyzer:
    def __init__(self):
        pass

    def calculate_plucking_vector(self, mask_points, image_shape):
        """
        Takes a YOLO polygon mask and calculates the centroid and angle.
        mask_points: List of [x, y] coordinates from YOLO.
        image_shape: (height, width) of the original image.
        """
        if len(mask_points) < 5:
            return None # Not enough points to calculate geometry
            
        # Convert normalized YOLO coordinates back to pixel coordinates
        h, w = image_shape[:2]
        pixel_points = np.array([[int(p[0] * w), int(p[1] * h)] for p in mask_points], dtype=np.int32)
        
        # 1. Calculate the Centroid (Center of Mass) using Image Moments
        M = cv2.moments(pixel_points)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0
            
        # 2. Calculate the Angle of Orientation
        # Fit an ellipse to the polygon to find its main axis
        try:
            (x, y), (MA, ma), angle = cv2.fitEllipse(pixel_points)
        except cv2.error:
            angle = 0.0 # Fallback if ellipse fitting fails on a weird shape

        return {
            "centroid_x": cX,
            "centroid_y": cY,
            "angle_degrees": round(angle, 2)
        }

    def draw_targeting_reticle(self, image, vector_data):
        """Visually draws the calculated plucking point on the image."""
        if not vector_data:
            return image
            
        cX = vector_data["centroid_x"]
        cY = vector_data["centroid_y"]
        
        # Draw a bright green target at the centroid
        cv2.circle(image, (cX, cY), 5, (0, 255, 0), -1)
        cv2.line(image, (cX - 15, cY), (cX + 15, cY), (0, 255, 0), 2)
        cv2.line(image, (cX, cY - 15), (cX, cY + 15), (0, 255, 0), 2)
        
        # Put the coordinates as text
        text = f"TARGET: ({cX}, {cY}) | Ang: {vector_data['angle_degrees']}*"
        cv2.putText(image, text, (cX + 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return image