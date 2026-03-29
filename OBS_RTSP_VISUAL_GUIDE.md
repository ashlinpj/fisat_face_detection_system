# 📸 OBS RTSP Setup - Step-by-Step Visual Guide

## 🎯 Goal
Connect your face detection system to OBS video stream using RTSP protocol.

---

## 📥 Step 1: Download and Install

### 1.1 Install OBS Studio
```
1. Go to: https://obsproject.com/download
2. Download OBS Studio for Windows
3. Run the installer
4. Follow installation wizard
5. Launch OBS Studio
```

### 1.2 Install RTSP Server Plugin
```
1. Go to: https://github.com/iamscottxu/obs-rtspserver/releases
2. Download: obs-rtspserver-v3.x.x-windows-installer.exe
3. Close OBS if running
4. Run the plugin installer
5. Click "Install"
6. Restart OBS Studio
```

---

## 🎥 Step 2: Configure OBS

### 2.1 Add Your Camera Source

```
In OBS Studio:

1. Look at the "Sources" panel (bottom middle)
2. Click the [+] button
3. Select "Video Capture Device"
4. Click "OK" in the dialog
5. Name it: "My Camera" (or any name)
6. Click "OK"

7. In the next window:
   - Device: Select your webcam
   - Resolution: 640x480 (recommended)
   - FPS: 30
   - Click "OK"

8. You should now see your camera feed in OBS
```

### 2.2 Configure RTSP Server

```
In OBS Studio:

1. Click "Tools" in the top menu
2. Select "RTSP Server Settings"

3. In the RTSP Settings window:
   ☑ Enable RTSP Server           [Check this box]
   
   Server Port: 8554               [Default, don't change]
   
   URL Suffix: /live               [Use exactly: /live]
   
   ☐ Enable Authentication         [Leave unchecked for now]
   
   ☑ Auto Start                    [Check this box]
   
4. Click "OK"

5. Your RTSP URL is now: rtsp://127.0.0.1:8554/live
```

### 2.3 Start Streaming

```
In OBS Studio:

1. Look at the bottom-right controls
2. Click "Start Streaming" button
3. The button should change to "Stop Streaming"
4. Your RTSP server is now broadcasting!

Status indicator:
🔴 Not streaming → Click "Start Streaming"
🟢 Streaming    → Ready for face detection!
```

---

## ⚙️ Step 3: Configure Your Python Application

### 3.1 Edit config.py

```python
Open: config.py (in your project folder)

Find this line:
    USE_RTSP = False

Change to:
    USE_RTSP = True

Find this line:
    RTSP_URL = "rtsp://127.0.0.1:8554/live"

Make sure it says:
    RTSP_URL = "rtsp://127.0.0.1:8554/live"

Save the file!
```

### 3.2 File Location

```
Your project folder:
fisat_face_detection_system/
  ├── config.py           ← Edit this file
  ├── main.py
  ├── test_rtsp.py
  └── ...
```

---

## 🧪 Step 4: Test the Connection

### Option A: Using test_rtsp.py (Recommended)

```bash
# Open terminal in your project folder
# Press Win+R, type: cmd, press Enter
# Navigate to project folder:
cd "f:\VS Code Saves\FaceProject\new_face_id\fisat_face_detection_system"

# Run the test:
python test_rtsp.py

# Follow the prompts:
1. Choose option 1 (RTSP Stream)
2. Press Enter to use default URL
3. Wait for test results

Expected output:
✓ Connection established!
✓ Successfully reading frames!
✓ RTSP stream is working well!
```

### Option B: Using VLC Player

```
1. Open VLC Media Player
2. Click "Media" → "Open Network Stream"
3. Enter this URL: rtsp://127.0.0.1:8554/live
4. Click "Play"

If you see your camera feed → RTSP is working! ✓
```

---

## 🚀 Step 5: Run Your Application

### Start Face Detection

```bash
# In terminal:
python main.py

# Or double-click:
main.py

# You'll see:
Connecting to RTSP stream: rtsp://127.0.0.1:8554/live
✓ RTSP stream connected successfully!

# Then select option 1:
Enter choice (1-6): 1

# Face detection window will open with RTSP stream!
```

---

## 📋 Quick Reference Card

```
╔════════════════════════════════════════════════════════╗
║  OBS RTSP Quick Reference                              ║
╠════════════════════════════════════════════════════════╣
║  RTSP URL:    rtsp://127.0.0.1:8554/live               ║
║  Port:        8554                                     ║
║  URL Suffix:  /live                                    ║
║  Transport:   TCP (recommended)                        ║
╠════════════════════════════════════════════════════════╣
║  OBS Settings Location:                                ║
║    Tools → RTSP Server Settings                        ║
║                                                        ║
║  Start Streaming:                                      ║
║    Bottom-right "Start Streaming" button               ║
║                                                        ║
║  Test Connection:                                      ║
║    VLC → Media → Open Network Stream                   ║
║                                                        ║
║  Python Config:                                        ║
║    config.py → USE_RTSP = True                         ║
╚════════════════════════════════════════════════════════╝
```

---

## 🔧 Settings Optimization

### For Best Performance:

```
In OBS Studio:

1. Right-click on your camera source
2. Click "Properties"
3. Set:
   Resolution: 640x480
   FPS: 30

4. Go to Settings → Video
   Base Resolution: 640x480
   Output Resolution: 640x480
   FPS: 30

5. Go to Settings → Output
   Streaming:
     Encoder: x264
     Bitrate: 2500
```

### In config.py:

```python
# For lowest latency:
RTSP_BUFFER_SIZE = 1
RTSP_TRANSPORT = "tcp"

# For stability:
RTSP_RECONNECT_ATTEMPTS = 5
RTSP_RECONNECT_DELAY = 2
```

---

## ❌ Common Issues & Solutions

### Issue 1: "Could not connect to RTSP stream"

```
Checklist:
☐ Is OBS running?
☐ Did you click "Start Streaming" in OBS?
☐ Is RTSP Server plugin installed?
☐ Is RTSP Server enabled in Tools → RTSP Server Settings?
☐ Is the URL exactly: rtsp://127.0.0.1:8554/live ?
☐ Check Windows Firewall → Allow port 8554

Solution:
1. Restart OBS
2. Verify RTSP settings
3. Test with VLC first
4. Check firewall
```

### Issue 2: OBS doesn't show RTSP Server Settings

```
Problem: RTSP plugin not installed correctly

Solution:
1. Close OBS completely
2. Reinstall RTSP Server plugin
3. Restart OBS
4. Check Tools menu again
```

### Issue 3: Black screen in face detection

```
Problem: OBS not streaming or wrong source

Solution:
1. In OBS, verify camera is visible
2. Click "Start Streaming"
3. Test in VLC first
4. Restart Python application
```

### Issue 4: High latency (delay)

```
Problem: Network buffering

Solution in config.py:
RTSP_BUFFER_SIZE = 1  # Lower value

In OBS:
- Reduce resolution to 640x480
- Set FPS to 30
- Use wired connection (not WiFi)
```

---

## 🎓 Understanding RTSP URLs

```
Format: rtsp://[username:password@]host:port/path

Examples:

1. Local OBS (no password):
   rtsp://127.0.0.1:8554/live
   
2. Local OBS (with password):
   rtsp://admin:mypass@127.0.0.1:8554/live
   
3. Remote OBS (another PC):
   rtsp://192.168.1.100:8554/live
   
4. IP Camera:
   rtsp://admin:admin@192.168.1.64:554/stream
   
5. Network camera:
   rtsp://user:pass@camera.local:554/live/ch1

Where:
- 127.0.0.1 = Your own computer
- 192.168.x.x = Another PC on your network
- Port 8554 = OBS default
- Port 554 = IP camera default
- /live = URL suffix (set in OBS)
```

---

## ✅ Final Checklist

Before starting face detection:

```
OBS Setup:
☐ OBS Studio installed
☐ RTSP Server plugin installed
☐ Camera added to Sources
☐ RTSP Server enabled (Tools → RTSP Server Settings)
☐ Port: 8554, URL Suffix: /live
☐ "Start Streaming" button clicked
☐ Camera feed visible in OBS

Python Setup:
☐ config.py edited
☐ USE_RTSP = True
☐ RTSP_URL = "rtsp://127.0.0.1:8554/live"
☐ File saved

Testing:
☐ Tested with VLC (optional)
☐ Tested with python test_rtsp.py
☐ Got "✓ RTSP stream is working well!" message

Ready to Run:
☐ python main.py
☐ Select option 1
☐ Enjoy face detection! 🎉
```

---

## 🆘 Getting Help

1. **Test RTSP:** `python test_rtsp.py`
2. **Test in VLC:** Verify OBS is streaming
3. **Check OBS:** Ensure "Start Streaming" is active
4. **Check config.py:** Verify RTSP settings
5. **Firewall:** Allow port 8554

---

## 📱 Quick Commands

```bash
# Test connection
python test_rtsp.py

# Run face detection
python main.py

# Quick config (Windows)
rtsp_config.bat

# Switch to webcam (edit config.py)
USE_RTSP = False

# Switch to RTSP (edit config.py)
USE_RTSP = True
```

---

**That's it! You're ready to use RTSP streaming with face detection!** 🎉

Remember: Keep OBS running and streaming whenever you want to use RTSP mode.
