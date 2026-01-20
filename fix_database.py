"""
Database Recovery Script
Scans the 'faces' folder and adds missing students to the database.
"""
import os
import sqlite3
import config
import database

def fix_system():
    print("🔧 STARTING DATABASE RECOVERY...")
    
    # 1. Initialize DB
    database.init_database()
    
    # 2. Scan Faces Folder
    if not os.path.exists(config.FACES_DIR):
        print("❌ Error: Faces folder not found!")
        return

    files = os.listdir(config.FACES_DIR)
    print(f"📂 Found {len(files)} images in faces folder.")

    # 3. Group images by Student ID
    # Filenames are: STUDENTID_source_timestamp.jpg
    found_students = {}
    
    for filename in files:
        if not filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue
            
        try:
            # Extract ID (everything before the first underscore)
            student_id = filename.split('_')[0]
            
            if student_id not in found_students:
                found_students[student_id] = os.path.join(config.FACES_DIR, filename)
        except:
            continue

    print(f"🧐 Identified {len(found_students)} unique Student IDs from files.\n")

    # 4. Check DB and Insert Missing
    for student_id, image_path in found_students.items():
        # Check if exists
        existing = database.get_student_by_id(student_id)
        
        if existing:
            print(f"✅ {existing['name']} ({student_id}) is already in DB.")
        else:
            print(f"⚠️  MISSING FROM DB: ID {student_id}")
            print(f"   (We found photos but no database entry)")
            
            # Ask user for details to recover the record
            name = input(f"   >> Enter Name for {student_id}: ")
            dept = input(f"   >> Enter Dept for {student_id} (e.g. CSE): ")
            
            # Add to Database
            success = database.add_student(
                student_id=student_id,
                name=name,
                department=dept,
                year=1,  # Default to 1 to save time
                face_embedding=None,
                face_image_path=image_path
            )
            
            if success:
                print(f"   🎉 Recovered {name} into Database!\n")
            else:
                print(f"   ❌ Failed to add {student_id}.\n")

    print("✅ RECOVERY COMPLETE.")
    print("👉 Now open the App and click 'Train System'!")

if __name__ == "__main__":
    fix_system()