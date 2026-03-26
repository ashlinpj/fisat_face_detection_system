"""
RTSP Stream Tester
Simple script to test RTSP stream connection before running the full application
"""

import cv2
import sys
import time


def test_rtsp_stream(rtsp_url):
    """Test RTSP stream connection"""
    print("=" * 60)
    print("  RTSP STREAM TESTER")
    print("=" * 60)
    print(f"\nTesting connection to: {rtsp_url}")
    print("Please wait...")

    # Try to connect
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        print("\n✗ Failed to connect to RTSP stream!")
        print("\nPossible issues:")
        print("  1. RTSP server is not running")
        print("  2. Wrong RTSP URL")
        print("  3. Firewall blocking connection")
        print("  4. Network issue")
        return False

    # Try to read a frame
    print("✓ Connection established!")
    print("\nTrying to read frames...")

    frame_count = 0
    start_time = time.time()

    for i in range(30):  # Test 30 frames
        ret, frame = cap.read()
        if ret:
            frame_count += 1
            if frame_count == 1:
                height, width = frame.shape[:2]
                print("✓ Successfully reading frames!")
                print(f"  Resolution: {width}x{height}")
        else:
            print(f"✗ Failed to read frame {i+1}")

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    elapsed = time.time() - start_time
    fps = frame_count / elapsed if elapsed > 0 else 0

    print("\nResults:")
    print(f"  Frames read: {frame_count}/30")
    print(f"  Average FPS: {fps:.2f}")

    cap.release()

    if frame_count >= 25:
        print("\n✓ RTSP stream is working well!")
        print("  You can use this stream in your application.")
        return True
    elif frame_count > 0:
        print("\n⚠ Stream is working but unstable")
        print("  Consider:")
        print("    - Improving network connection")
        print("    - Reducing stream quality")
        print("    - Using TCP instead of UDP")
        return True
    else:
        print("\n✗ Could not read any frames")
        return False


def test_webcam(camera_index=0):
    """Test webcam connection"""
    print("=" * 60)
    print("  WEBCAM TESTER")
    print("=" * 60)
    print(f"\nTesting camera index: {camera_index}")

    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print("\n✗ Failed to open webcam!")
        print(f"  Camera index {camera_index} not available")
        return False

    ret, frame = cap.read()
    if ret:
        height, width = frame.shape[:2]
        print("✓ Webcam opened successfully!")
        print(f"  Resolution: {width}x{height}")
        cap.release()
        return True
    else:
        print("\n✗ Could not read from webcam")
        cap.release()
        return False


def main():
    print("\nWhat would you like to test?")
    print("  1. RTSP Stream")
    print("  2. Webcam")
    print("  3. Both")

    try:
        choice = input("\nEnter choice (1-3): ").strip()

        if choice == "1":
            rtsp_url = input("\nEnter RTSP URL [rtsp://127.0.0.1:8554/live]: ").strip()
            if not rtsp_url:
                rtsp_url = "rtsp://127.0.0.1:8554/live"
            test_rtsp_stream(rtsp_url)

        elif choice == "2":
            camera_index = input("\nEnter camera index [0]: ").strip()
            camera_index = int(camera_index) if camera_index else 0
            test_webcam(camera_index)

        elif choice == "3":
            # Test webcam first
            print("\n--- Testing Webcam ---")
            test_webcam(0)

            print("\n" + "=" * 60)
            input("\nPress Enter to test RTSP stream...")

            # Test RTSP
            print("\n--- Testing RTSP Stream ---")
            rtsp_url = input("\nEnter RTSP URL [rtsp://127.0.0.1:8554/live]: ").strip()
            if not rtsp_url:
                rtsp_url = "rtsp://127.0.0.1:8554/live"
            test_rtsp_stream(rtsp_url)

        else:
            print("Invalid choice")

    except KeyboardInterrupt:
        print("\n\nTest cancelled by user")
    except Exception as e:
        print(f"\nError: {e}")


if __name__ == "__main__":
    main()
