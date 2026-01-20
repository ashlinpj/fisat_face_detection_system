import sqlite3
import config
import os

def clear_history():
    if not os.path.exists(config.DATABASE_PATH):
        print("Database not found!")
        return

    conn = sqlite3.connect(config.DATABASE_PATH)
    cursor = conn.cursor()
    
    # Check how many logs exist
    cursor.execute("SELECT COUNT(*) FROM visit_logs")
    count = cursor.fetchone()[0]
    
    print(f"📊 Found {count} entries in Visit Logs.")
    
    if count > 0:
        confirm = input("⚠️  Are you sure you want to DELETE ALL visit history? (yes/no): ")
        if confirm.lower() == "yes":
            cursor.execute("DELETE FROM visit_logs")
            # Reset the ID counter for logs back to 1
            cursor.execute("DELETE FROM sqlite_sequence WHERE name='visit_logs'")
            conn.commit()
            print("✅ Success! All visit logs cleared.")
            print("   (Student registrations were NOT touched)")
        else:
            print("❌ Operation cancelled.")
    else:
        print("   Nothing to delete.")

    conn.close()

if __name__ == "__main__":
    clear_history()