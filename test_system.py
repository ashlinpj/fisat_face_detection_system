"""
Test Script - IMPROVED VERSION
Register faces and test with webcam - better accuracy
"""

import cv2
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))      

import config
import database
from face_recognition_module import FaceRecognitionSystem

def register_from_image(system, image_path, student_id, name, department="CSE", year=3):
    """Register a student from an image file"""
    print(f"\nRegistering from image: {image_path}")
    
    if not os.path.exists(image_path):
        print(f"  ✗ Error: Image not found: {image_path}")
        return False
    
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"  ✗ Error: Could not read image")
        return False
    
    success = system.register_new_student(frame, student_id, name, department, year)
    return success

def register_from_webcam(system, student_id, name, department="CSE", year=3):
    """Register directly from webcam with multiple captures for better accuracy"""
    print(f"\nRegistering from webcam: {name} ({student_id})")
    
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("  ✗ Could not open webcam!")
        return False
    
    print("  Look at camera. Press SPACE to capture, Q to cancel")
    
    captured_frame = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Show preview
        preview = frame.copy()
        cv2.putText(preview, "Press SPACE to capture, Q to cancel", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(preview, f"Registering: {name}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.imshow("Registration", preview)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            captured_frame = frame.copy()
            break
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    if captured_frame is not None:
        success = system.register_new_student(captured_frame, student_id, name, department, year)
        return success
    
    return False

def register_all_from_folder(system, folder_path):
    """
    Register all images from a folder
    Image filename format: StudentID_Name.jpg (e.g., 21CS001_John.jpg)
    """
    print(f"\nScanning folder: {folder_path}")
    
    if not os.path.exists(folder_path):
        print(f"  ✗ Folder not found!")
        return
    
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    registered = 0
    
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(image_extensions):
            filepath = os.path.join(folder_path, filename)
            
            # Parse filename: StudentID_Name.jpg
            name_part = os.path.splitext(filename)[0]
            parts = name_part.split('_')
            
            if len(parts) >= 2:
                student_id = parts[0]
                name = '_'.join(parts[1:]).replace('_', ' ')
            else:
                student_id = f"STU{registered+1:03d}"
                name = name_part
            
            if register_from_image(system, filepath, student_id, name):
                registered += 1
    
    print(f"\n✓ Registered {registered} students from folder")

def test_with_webcam(system):
    """Test face recognition with webcam - optimized"""
    print("\n" + "=" * 50)
    print("  WEBCAM TEST MODE (Optimized)")
    print("=" * 50)
    print("\nControls:")
    print("  [Q] - Quit")
    print("  [R] - Register current face")
    print("  [S] - Show registered students")
    print("-" * 50)
    
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 60)  # Request high FPS
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency
    
    if not cap.isOpened():
        print("✗ Error: Could not open webcam!")
        return
    
    print("✓ Webcam opened. Press Q to quit.\n")
    
    # FPS counter
    fps_start = time.time()
    fps_count = 0
    fps = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        annotated_frame, recognized = system.process_frame(frame)
        
        # Calculate FPS
        fps_count += 1
        elapsed = time.time() - fps_start
        if elapsed >= 1.0:
            fps = int(fps_count / elapsed)
            fps_count = 0
            fps_start = time.time()
        
        # Add FPS display (top-right with color based on performance)
        fps_color = (0, 255, 0) if fps >= 25 else (0, 165, 255) if fps >= 15 else (0, 0, 255)
        cv2.putText(annotated_frame, f"FPS: {fps}", (annotated_frame.shape[1] - 100, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, fps_color, 2)
        
        # Display
        cv2.imshow("Face Detection Test - Q:Quit R:Register S:Students", annotated_frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == ord('Q'):
            break
        
        elif key == ord('r') or key == ord('R'):
            # Pause video and get input
            cv2.imshow("Face Detection Test", frame)
            print("\n--- Quick Registration ---")
            student_id = input("Enter Student ID: ").strip()
            name = input("Enter Name: ").strip()
            
            if student_id and name:
                ret, capture_frame = cap.read()
                if ret:
                    if system.register_new_student(capture_frame, student_id, name, "CSE", 3):
                        print("✓ Registration successful!")
                    else:
                        print("✗ Registration failed!")
        
        elif key == ord('s') or key == ord('S'):
            students = database.get_all_students()
            print(f"\n--- Registered Students ({len(students)}) ---")
            for s in students:
                print(f"  {s['student_id']}: {s['name']}")
            print()
    
    cap.release()
    cv2.destroyAllWindows()

def main():
    print("=" * 60)
    print("  FACE DETECTION SYSTEM - TEST MODE (Optimized)")
    print("=" * 60)
    
    # Initialize
    print("\nInitializing system...")
    database.init_database()
    system = FaceRecognitionSystem()
    print("✓ System ready!\n")
    
    while True:
        print("\n" + "-" * 40)
        print("Options:")
        print("  1. Register from IMAGE FILE")
        print("  2. Register ALL from FOLDER")
        print("  3. Register from WEBCAM (recommended)")
        print("  4. Test with WEBCAM")
        print("  5. View registered students")
        print("  6. Re-register (update face)")
        print("  7. Exit")
        print("-" * 40)
        
        choice = input("Enter choice (1-7): ").strip()
        
        if choice == '1':
            print("\n--- Register from Image ---")
            image_path = input("Enter image path: ").strip().strip('"').strip("'")
            student_id = input("Enter Student ID: ").strip()
            name = input("Enter Name: ").strip()
            department = input("Enter Department (default: CSE): ").strip() or "CSE"
            year = input("Enter Year 1-4 (default: 3): ").strip() or "3"
            
            register_from_image(system, image_path, student_id, name, department, int(year))
        
        elif choice == '2':
            print("\n--- Register from Folder ---")
            print("Image filename format: StudentID_Name.jpg")
            folder_path = input("\nEnter folder path: ").strip().strip('"').strip("'")
            register_all_from_folder(system, folder_path)
        
        elif choice == '3':
            print("\n--- Register from Webcam ---")
            student_id = input("Enter Student ID: ").strip()
            name = input("Enter Name: ").strip()
            department = input("Enter Department (default: CSE): ").strip() or "CSE"
            year = input("Enter Year 1-4 (default: 3): ").strip() or "3"
            
            if student_id and name:
                register_from_webcam(system, student_id, name, department, int(year))
        
        elif choice == '4':
            test_with_webcam(system)
        
        elif choice == '5':
            students = database.get_all_students()
            print(f"\n--- Registered Students ({len(students)}) ---")
            if students:
                for s in students:
                    print(f"  {s['student_id']}: {s['name']} ({s.get('department', 'N/A')}, Year {s.get('year', 'N/A')})")
            else:
                print("  No students registered yet.")
        
        elif choice == '6':
            print("\n--- Re-register (Update Face) ---")
            students = database.get_all_students()
            print("Current students:")
            for s in students:
                print(f"  {s['student_id']}: {s['name']}")
            
            student_id = input("\nEnter Student ID to update: ").strip()
            existing = database.get_student_by_id(student_id)
            if existing:
                print(f"Updating: {existing['name']}")
                register_from_webcam(system, student_id, existing['name'], 
                                    existing.get('department', 'CSE'), existing.get('year', 3))
            else:
                print("Student not found!")
        
        elif choice == '7':
            print("\nGoodbye!")
            break

if __name__ == "__main__":
    main()
