"""
Setup Script - Install dependencies and initialize the system
Run this first before using the application
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("=" * 60)
    print("  COLLEGE CANTEEN FACE DETECTION SYSTEM - SETUP")
    print("=" * 60)
    
    print("\n[1/4] Installing required packages...")
    print("This may take a few minutes...\n")
    
    # Install packages
    packages = [
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "Pillow>=10.0.0",
        "ultralytics>=8.1.0",
        "deepface>=0.0.79",
        "tensorflow>=2.13.0"
    ]
    
    for package in packages:
        print(f"Installing {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])
            print(f"  ✓ {package} installed")
        except subprocess.CalledProcessError as e:
            print(f"  ✗ Failed to install {package}: {e}")
    
    print("\n[2/4] Creating directories...")
    
    # Create directories
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dirs = [
        os.path.join(base_dir, "database", "faces"),
        os.path.join(base_dir, "screenshots")
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"  ✓ Created: {dir_path}")
    
    print("\n[3/4] Initializing database...")
    
    try:
        from app.repositories.connection_pool import init_database
        init_database()
        print("  ✓ Database initialized")
    except Exception as e:
        print(f"  ✗ Database initialization failed: {e}")
    
    print("\n[4/4] Downloading YOLO model...")
    print("  (This will download automatically on first run)")
    
    print("\n" + "=" * 60)
    print("  SETUP COMPLETE!")
    print("=" * 60)
    print("\nYou can now run the application:")
    print("  - GUI Mode:     python gui.py")
    print("  - Console Mode: python main.py")
    print("\nFirst run will download YOLO and DeepFace models (~500MB)")
    print("=" * 60)

if __name__ == "__main__":
    install_requirements()
