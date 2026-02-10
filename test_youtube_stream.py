"""
Quick test script to verify YouTube stream functionality
"""

import cv2
import sys

def test_youtube_stream(youtube_url):
    """Test if we can extract and read from a YouTube stream"""
    try:
        import yt_dlp
        
        print(f"Testing YouTube URL: {youtube_url}")
        print("=" * 60)
        
        # Extract stream URL
        print("\n[1/3] Extracting stream URL from YouTube...")
        ydl_opts = {
            'format': 'best[height<=720]',
            'quiet': True,
            'no_warnings': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
            stream_url = info['url']
            
        print(f"✓ Stream extracted successfully!")
        print(f"  Title: {info.get('title', 'Unknown')}")
        print(f"  Duration: {info.get('duration', 'Unknown')} seconds")
        
        # Test OpenCV capture
        print("\n[2/3] Opening stream with OpenCV...")
        cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
        
        if not cap.isOpened():
            print("✗ Failed to open stream with OpenCV")
            return False
        
        print("✓ Stream opened successfully!")
        
        # Test reading frames
        print("\n[3/3] Testing frame reading...")
        ret, frame = cap.read()
        
        if not ret:
            print("✗ Failed to read frame from stream")
            cap.release()
            return False
        
        print("✓ Successfully read frame from stream!")
        print(f"  Frame shape: {frame.shape}")
        print(f"  Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
        
        # Clean up
        cap.release()
        
        print("\n" + "=" * 60)
        print("✓ YouTube stream test PASSED!")
        print("=" * 60)
        print("\nYou can now use this URL in config.py")
        return True
        
    except ImportError:
        print("\n✗ ERROR: yt-dlp is not installed!")
        print("Install it with: pip install yt-dlp")
        return False
    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}")
        return False

if __name__ == "__main__":
    # Example YouTube URLs to test
    print("YouTube Stream Test Utility")
    print("=" * 60)
    
    if len(sys.argv) > 1:
        youtube_url = sys.argv[1]
    else:
        # Default test URL - lofi hip hop radio (24/7 live stream)
        youtube_url = "https://www.youtube.com/watch?v=jfKfPfyJRdk"
        print(f"\nNo URL provided. Using default test stream:")
        print(f"  {youtube_url}")
        print("\nUsage: python test_youtube_stream.py <youtube_url>")
        print()
    
    test_youtube_stream(youtube_url)
