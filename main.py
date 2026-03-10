"""Main Application - College Canteen Face Detection System

Thin launcher that creates a DI Container and delegates to the CLI app.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.utils.logging_config import setup_logging
from app.container import Container
from app.ui.cli_app import CanteenFaceDetectionApp


def main():
    """Main entry point"""
    setup_logging()
    container = Container()
    app = CanteenFaceDetectionApp(container)

    print("\nOptions:")
    print("  1. Start Real-time Detection")
    print("  2. Register New Student")
    print("  3. View Statistics")
    print("  4. View Today's Logs")
    print("  5. List All Students")
    print("  6. Exit")

    while True:
        try:
            choice = input("\nEnter choice (1-6): ").strip()

            if choice == '1':
                app.run_detection()
            elif choice == '2':
                if app.start_camera():
                    ret, frame = app.cap.read()
                    if ret:
                        app.register_student_interactive(frame)
                    app.stop_camera()
            elif choice == '3':
                app.show_statistics()
            elif choice == '4':
                app.show_todays_logs()
            elif choice == '5':
                app.list_students()
            elif choice == '6':
                print("\nGoodbye!")
                break
            else:
                print("Invalid choice. Please enter 1-6.")

        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

    # Final cleanup
    if hasattr(app, 'cap') and app.cap:
        app.cleanup_camera()
    container.stop()


if __name__ == "__main__":
    main()
