"""
Utility functions for the Face Detection System
Handles Reporting, Drawing, and Data Augmentation
"""

import os
import cv2
import numpy as np
from datetime import datetime, timedelta
from typing import List, Tuple
from PIL import Image
import database
import config

# --- PART 1: DATA AUGMENTATION (New for Training) ---

def generate_augmented_images(image_path: str, num_variations: int = 10) -> List[np.ndarray]:
    """
    Reads an image and returns a list of augmented versions 
    (Original + Blurred + Rotated + Brightness adjusted).
    Used by train_model.py to create robust embeddings.
    """
    images = []
    
    # 1. Load Original
    img_cv = cv2.imread(image_path)
    if img_cv is None: 
        return []
    
    # Add Original
    images.append(img_cv)
    
    try:
        # Import only when needed to save startup time if not training
        from torchvision import transforms
        from PIL import Image
        import torch

        # Convert to RGB for Pillow/Torch
        img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        
        # 2. Define Augmentations (The "Messy" conditions)
        # We want the AI to recognize the person even if the camera is bad
        transform_pipeline = transforms.Compose([
            transforms.RandomRotation(degrees=15),    # Head tilt
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2), # Bad lighting
            transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2.0)), # Blur/Motion
            transforms.ToTensor()
        ])

        for _ in range(num_variations):
            try:
                # Apply transforms
                aug_tensor = transform_pipeline(img_pil)
                # Convert back to OpenCV format (BGR)
                aug_pil = transforms.ToPILImage()(aug_tensor)
                aug_cv = cv2.cvtColor(np.array(aug_pil), cv2.COLOR_RGB2BGR)
                images.append(aug_cv)
            except Exception as e:
                pass

    except ImportError:
        print("⚠️ Torch/Torchvision not found. Using simple OpenCV augmentation.")
        # Fallback if Torch is missing
        rows, cols, _ = img_cv.shape
        for _ in range(num_variations):
            # Simple Blur
            blur = cv2.GaussianBlur(img_cv, (5, 5), 0)
            images.append(blur)
            # Simple Rotation
            M = cv2.getRotationMatrix2D((cols/2, rows/2), np.random.randint(-10, 10), 1)
            dst = cv2.warpAffine(img_cv, M, (cols, rows))
            images.append(dst)

    return images

# --- PART 2: REPORTING & EXPORTS (Existing) ---

def generate_report(start_date: str = None, end_date: str = None) -> str:
    """Generate a detailed report for the specified date range"""
    if start_date is None:
        start_date = datetime.now().strftime('%Y-%m-%d')
    if end_date is None:
        end_date = start_date
    
    report = []
    report.append("=" * 60)
    report.append("COLLEGE CANTEEN FACE DETECTION SYSTEM - REPORT")
    report.append("=" * 60)
    report.append(f"Report Period: {start_date} to {end_date}")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("-" * 60)
    
    # Get statistics
    stats = database.get_daily_statistics(start_date)
    
    report.append("\nDAILY SUMMARY")
    report.append(f"  Total Visits: {stats['total_visits']}")
    report.append(f"  Unique Visitors: {stats['unique_visitors']}")
    report.append(f"  Unknown Visitors: {stats['unknown_visitors']}")
    report.append(f"  Average Duration: {stats['average_duration_minutes']} minutes")
    
    # Get all students
    students = database.get_all_students()
    report.append(f"\nREGISTERED STUDENTS: {len(students)}")
    
    # Get visit logs
    logs = database.get_visit_logs(date=start_date)
    
    report.append(f"\nVISIT LOG ({len(logs)} entries)")
    report.append("-" * 60)
    report.append(f"{'Time':<12} {'Student ID':<15} {'Name':<25} {'Status':<10}")
    report.append("-" * 60)
    
    for log in logs:
        entry_time = log.get('entry_time', 'N/A')
        if entry_time and 'T' in entry_time:
            entry_time = entry_time.split('T')[1][:8]
        elif entry_time and ' ' in entry_time:
            entry_time = entry_time.split(' ')[1][:8]
        
        student_id = str(log.get('student_id', 'Unknown'))[:13]
        name = str(log.get('student_name', 'Unknown'))[:23]
        status = "Known" if log.get('is_known') else "Unknown"
        
        report.append(f"{entry_time:<12} {student_id:<15} {name:<25} {status:<10}")
    
    report.append("=" * 60)
    
    return "\n".join(report)

def export_report_to_file(filepath: str, start_date: str = None, end_date: str = None):
    """Export report to a text file"""
    report = generate_report(start_date, end_date)
    try:
        with open(filepath, 'w') as f:
            f.write(report)
        print(f"Report exported to: {filepath}")
    except Exception as e:
        print(f"Error exporting report: {e}")

def export_logs_to_csv(filepath: str, date: str = None, student_id: str = None):
    """Export visit logs to CSV file"""
    logs = database.get_visit_logs(date=date, student_id=student_id)
    
    try:
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            f.write("ID,Date,Entry Time,Exit Time,Student ID,Student Name,Duration (min),Status\n")
            
            for log in logs:
                status = "Known" if log.get('is_known') else "Unknown"
                duration = log.get('duration_minutes', '')
                
                f.write(f"{log['id']},"
                       f"{log.get('date', '')},"
                       f"{log.get('entry_time', '')},"
                       f"{log.get('exit_time', '')},"
                       f"{log.get('student_id', '')},"
                       f"\"{log.get('student_name', '')}\","
                       f"{duration},"
                       f"{status}\n")
        print(f"Logs exported to: {filepath}")
    except Exception as e:
        print(f"Error exporting logs: {e}")

def export_students_to_csv(filepath: str):
    """Export student list to CSV file"""
    students = database.get_all_students()
    
    try:
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            f.write("ID,Student ID,Name,Department,Year,Registered Date\n")
            
            for student in students:
                created = student.get('created_at', '')[:10] if student.get('created_at') else ''
                
                f.write(f"{student['id']},"
                       f"{student['student_id']},"
                       f"\"{student['name']}\","
                       f"{student.get('department', '')},"
                       f"{student.get('year', '')},"
                       f"{created}\n")
        print(f"Students exported to: {filepath}")
    except Exception as e:
        print(f"Error exporting students: {e}")

# --- PART 3: VISUALIZATION & ANALYTICS (Existing) ---

def get_hourly_distribution(date: str = None) -> dict:
    """Get hourly distribution of visits"""
    if date is None:
        date = datetime.now().strftime('%Y-%m-%d')
    
    logs = database.get_visit_logs(date=date)
    hourly = {i: 0 for i in range(24)}
    
    for log in logs:
        entry_time = log.get('entry_time', '')
        if entry_time:
            try:
                if 'T' in entry_time:
                    hour = int(entry_time.split('T')[1][:2])
                elif ' ' in entry_time:
                    hour = int(entry_time.split(' ')[1][:2])
                else:
                    continue
                hourly[hour] += 1
            except:
                pass
    return hourly

def get_peak_hours(date: str = None) -> List[Tuple[int, int]]:
    """Get peak hours sorted by visit count"""
    hourly = get_hourly_distribution(date)
    sorted_hours = sorted(hourly.items(), key=lambda x: x[1], reverse=True)
    return [(h, c) for h, c in sorted_hours if c > 0]

def draw_face_box(frame: np.ndarray, bbox: Tuple[int, int, int, int], 
                  name: str, confidence: float, is_known: bool) -> np.ndarray:
    """Draw a styled face bounding box with label"""
    x1, y1, x2, y2 = bbox
    
    # Colors
    if is_known:
        color = (0, 255, 0)  # Green for known
        bg_color = (0, 200, 0)
    else:
        color = (0, 0, 255)  # Red for unknown
        bg_color = (0, 0, 200)
    
    # Draw main rectangle
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    # Draw corner accents
    corner_length = 15
    thickness = 3
    
    # Corners
    cv2.line(frame, (x1, y1), (x1 + corner_length, y1), color, thickness)
    cv2.line(frame, (x1, y1), (x1, y1 + corner_length), color, thickness)
    cv2.line(frame, (x2, y1), (x2 - corner_length, y1), color, thickness)
    cv2.line(frame, (x2, y1), (x2, y1 + corner_length), color, thickness)
    cv2.line(frame, (x1, y2), (x1 + corner_length, y2), color, thickness)
    cv2.line(frame, (x1, y2), (x1, y2 - corner_length), color, thickness)
    cv2.line(frame, (x2, y2), (x2 - corner_length, y2), color, thickness)
    cv2.line(frame, (x2, y2), (x2, y2 - corner_length), color, thickness)
    
    # Label
    label = f"{name}"
    if confidence > 0:
        label += f" ({confidence:.0%})"
        
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 2
    
    (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
    
    # Label background
    cv2.rectangle(frame, 
                  (x1, y1 - text_height - 10), 
                  (x1 + text_width + 10, y1),
                  bg_color, -1)
    
    # Label text
    cv2.putText(frame, label, (x1 + 5, y1 - 5), 
                font, font_scale, (255, 255, 255), font_thickness)
    
    return frame

def resize_frame(frame: np.ndarray, max_width: int = 800) -> np.ndarray:
    """Resize frame maintaining aspect ratio"""
    h, w = frame.shape[:2]
    if w > max_width:
        scale = max_width / w
        new_w = max_width
        new_h = int(h * scale)
        return cv2.resize(frame, (new_w, new_h))
    return frame

def add_timestamp_overlay(frame: np.ndarray) -> np.ndarray:
    """Add timestamp overlay to frame"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.rectangle(frame, (5, 5), (260, 35), (0, 0, 0), -1)
    cv2.putText(frame, timestamp, (10, 28), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return frame

if __name__ == "__main__":
    # Test utilities
    print("Testing utilities...")
    database.init_database()
    print(generate_report())