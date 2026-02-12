# YouTube Stream Integration - Quick Guide

## ✅ What Was Done

The face detection system has been updated to support YouTube streams as a camera feed source.

### Changes Made:

1. **Installed yt-dlp** - Package for extracting YouTube stream URLs
2. **Updated config.py** - Added YouTube stream settings and disabled RTSP temporarily
3. **Updated main.py** - Added YouTube stream support in camera initialization
4. **Updated gui_app.py** - Added YouTube stream support in GUI application
5. **Updated requirements.txt** - Added yt-dlp dependency
6. **Created test_youtube_stream.py** - Test utility for verifying YouTube streams

### Current Configuration:

- ✅ **USE_YOUTUBE = True** (YouTube stream enabled)
- ❌ **USE_RTSP = False** (RTSP temporarily disabled)
- 🎵 Default stream: Lofi Hip Hop Radio (tested and working)

---

## 🚀 How to Use

### Option 1: Use Default Test Stream (Lofi Hip Hop Radio)

Simply run the application - it's already configured with a working 24/7 live stream:

```powershell
python gui_app.py
```

or

```powershell
python main.py
```

The default stream URL is: `https://www.youtube.com/watch?v=jfKfPfyJRdk`

### Option 2: Use Your Own YouTube Stream

1. **Edit config.py** and change the `YOUTUBE_URL`:

```python
YOUTUBE_URL = "https://www.youtube.com/watch?v=YOUR_VIDEO_ID"
```

2. **(Optional) Test the URL first**:

```powershell
python test_youtube_stream.py "https://www.youtube.com/watch?v=YOUR_VIDEO_ID"
```

3. **Run the application**:

```powershell
python gui_app.py
```

---

## 🔧 Configuration Options

In **config.py**, you can adjust these YouTube stream settings:

```python
# YouTube Stream settings
USE_YOUTUBE = True                    # Enable/disable YouTube stream
YOUTUBE_URL = "your_youtube_url"      # Your YouTube video/stream URL
YOUTUBE_RECONNECT_ATTEMPTS = 3        # Reconnection attempts if stream fails
YOUTUBE_RECONNECT_DELAY = 3           # Seconds between reconnection attempts
YOUTUBE_QUALITY = "best[height<=720]" # Stream quality
```

### Quality Options:

- `"best"` - Best available quality
- `"worst"` - Lowest quality (fastest)
- `"best[height<=720]"` - Best quality up to 720p (recommended)
- `"best[height<=480]"` - Best quality up to 480p (faster processing)

---

## 🔄 Switching Between Modes

### To use YouTube stream:

```python
USE_YOUTUBE = True
USE_RTSP = False
```

### To use RTSP stream:

```python
USE_YOUTUBE = False
USE_RTSP = True
```

### To use webcam:

```python
USE_YOUTUBE = False
USE_RTSP = False
```

---

## 🧪 Testing

Test any YouTube URL before using it:

```powershell
# Test default stream
python test_youtube_stream.py

# Test custom URL
python test_youtube_stream.py "https://www.youtube.com/watch?v=YOUR_ID"
```

The test will verify:

- ✓ Stream extraction from YouTube
- ✓ OpenCV can open the stream
- ✓ Frames can be read successfully

---

## 📝 Tips

1. **Live streams work best** - They provide continuous feed
2. **Shorter URLs are faster** - Live streams have less metadata to process
3. **Lower quality = faster processing** - If performance is slow, try `best[height<=480]`
4. **Check internet connection** - Streaming requires stable internet
5. **Age-restricted videos** - May not work (requires authentication)

---

## 🐛 Troubleshooting

### "Could not connect to YouTube stream"

- Check internet connection
- Verify the YouTube URL is accessible
- Try a different video/stream
- Test with: `python test_youtube_stream.py "your_url"`

### "Stream lost and reconnection failed"

- YouTube may have throttled the connection
- Try restarting the application
- Consider using a different video quality

### Slow/laggy performance

- Lower the quality in config.py: `YOUTUBE_QUALITY = "best[height<=480]"`
- Increase `PROCESS_EVERY_N_FRAMES` in config.py

---

## 🔙 Reverting to RTSP

To re-enable RTSP (original configuration):

1. Edit **config.py**:

```python
USE_YOUTUBE = False
USE_RTSP = True
```

2. Restart the application

---

## Example YouTube Streams to Try

- **Lofi Hip Hop Radio** (default): `https://www.youtube.com/watch?v=jfKfPfyJRdk`
- **NASA Live**: `https://www.youtube.com/watch?v=21X5lGlDOfg`
- **City Cams**: Search YouTube for "live city camera"

---

**Ready to test?** Just run:

```powershell
python gui_app.py
```

The application is now configured with a working YouTube stream! 🎉
