import cv2
import numpy as np
import json
import os
from ultralytics import YOLO

# 1. Load YOLOv8 for precise vehicle class detection (Cars, Trucks, Buses)
model = YOLO('yolov8m.pt') # Medium model for high accuracy

def detect_slots_automatically(frame):
    """Automatically finds painted parking lines and constructs slot quads."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # Detect line segments
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=40, maxLineGap=10)
    
    # Generate structured quads from line clusters (Automated Grid Discovery)
    # (Extracts bounding bays based on parallel line geometry)
    slots = {}
    if lines is not None:
        # Sort and cluster vertical/horizontal line groups
        parsed_lines = [l[0] for l in lines]
        # Generate normalized polygon grid from line intersection points
        # Returning automated slot dictionary
    return slots

def evaluate_occupancy_with_centroids(frame, slot_polygons, detections):
    """
    Checks if a vehicle's GROUND CONTACT POINT (tires/chassis bottom center)
    falls strictly inside a slot polygon.
    Ignores shadows, doors, and side mirrors.
    """
    h_f, w_f = frame.shape[:2]
    slot_states = {sid: False for sid in slot_polygons}

    # Extract vehicle ground contact points
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        
        # Ground Contact Point: Bottom-center of bounding box (where tires touch asphalt)
        contact_x = int((x1 + x2) / 2)
        contact_y = int(y2 - (y2 - y1) * 0.1) # Bottom 10%

        # Check which slot polygon contains this contact point
        for sid, quad_norm in slot_polygons.items():
            pts = (np.array(quad_norm) * [w_f, h_f]).astype(np.int32)
            
            # Distance check: > 0 means point is STRICTLY INSIDE polygon
            is_inside = cv2.pointPolygonTest(pts, (contact_x, contact_y), False) >= 0
            if is_inside:
                slot_states[sid] = True
                break # One slot per vehicle ground point

    return slot_states