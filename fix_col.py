import sqlite3
import config
import os

def fix_database():
    if not os.path.exists(config.DATABASE_PATH):
        print("❌ Database not found!")
        return

    print(f"🔧 Patching Database at: {config.DATABASE_PATH}")
    
    try:
        conn = sqlite3.connect(config.DATABASE_PATH)
        cursor = conn.cursor()
        
        # Add the missing column
        cursor.execute("ALTER TABLE visit_logs ADD COLUMN screenshot_path TEXT")
        
        conn.commit()
        conn.close()
        print("✅ Success! Added 'screenshot_path' column to visit_logs.")
        
    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e):
            print("⚠️ Column 'screenshot_path' already exists. No changes needed.")
        else:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    fix_database()