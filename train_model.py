"""
Training Script
Scans the faces directory, applies augmentation, 
calculates embeddings, and updates the SQLite database.
"""

import database
import config
import os
import cv2
import numpy as np
from deepface import DeepFace

# Fix: Import the correct function name from your current utils.py
from utils import generate_augmented_images

def train_system():
    print("🚀 Starting Multi-Image Training...")
    
    # 1. Initialize Database
    database.init_database()
    
    # 2. Get all registered students
    students = database.get_all_students()
    print(f"Found {len(students)} students in database.")
    
    # 3. Collect all available images for each student
    # Format: { '21CS001': ['path/to/img1.jpg', 'path/to/img2.jpg'] }
    student_images_map = {s['student_id']: [] for s in students}
    
    if os.path.exists(config.FACES_DIR):
        for filename in os.listdir(config.FACES_DIR):
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            
            # Match filename to student ID
            # Filenames are like: "21CS001_upload_12345.jpg" or "21CS001.jpg"
            for student_id in student_images_map.keys():
                if filename.startswith(student_id + "_") or filename == f"{student_id}.jpg":
                    full_path = os.path.join(config.FACES_DIR, filename)
                    student_images_map[student_id].append(full_path)
                    break

    # 4. Process each student
    for student in students:
        sid = student['student_id']
        name = student['name']
        images = student_images_map.get(sid, [])
        
        print(f"Processing {name} ({sid}): Found {len(images)} source images.")
        
        if not images:
            print("  ⚠️ No images found on disk. Skipping.")
            continue

        all_embeddings = []

        # 5. Generate embeddings for ALL real photos + their augmentations
        for img_path in images:
            # Generate variations (Blur, Rotate) for this specific photo
            # This calls the function in your utils.py
            variations = generate_augmented_images(img_path, num_variations=5)
            
            for face_img in variations:
                try:
                    # Get embedding using Facenet512
                    emb = DeepFace.represent(
                        img_path=face_img,
                        model_name="Facenet512",
                        enforce_detection=False,
                        detector_backend="skip"
                    )[0]["embedding"]
                    all_embeddings.append(emb)
                except Exception as e:
                    pass

        if not all_embeddings:
            print("  ❌ Could not extract features.")
            continue

        # 6. Average them into one Master Vector
        avg_embedding = np.mean(all_embeddings, axis=0)

        # 7. Save to DB
        database.update_student(
            student_id=sid,
            face_embedding=avg_embedding
        )
        print(f"  ✅ Updated with {len(all_embeddings)} total vectors (Real + Augmented).")

    print("\nTraining Complete!")

if __name__ == "__main__":
    train_system()