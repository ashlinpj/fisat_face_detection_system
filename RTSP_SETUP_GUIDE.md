# RTSP Setup Guide for Face Detection System

## Overview
This guide will help you set up RTSP streaming using OBS (Open Broadcaster Software) to stream your camera feed to the face detection system.

## What is RTSP?
RTSP (Real-Time Streaming Protocol) allows you to stream video over a network. This is useful for:
- Remote camera access
- Network cameras (IP cameras)
- Streaming from another computer
- Broadcasting camera feed to multiple applications

---

## Method 1: OBS + RTSP Server Plugin (Recommended)

### Step 1: Install OBS Studio
1. Download OBS Studio from: https://obsproject.com/download
2. Install OBS Studio on your computer
3. Launch OBS Studio

### Step 2: Install OBS RTSP Server Plugin

#### For Windows:
1. Download the plugin from: https://github.com/iamscottxu/obs-rtspserver/releases
   - Download `obs-rtspserver-v3.x.x-windows-installer.exe`
2. Close OBS if it's running
3. Run the installer
4. Accept the license and install
5. Restart OBS Studio

#### For Linux:
```bash
sudo add-apt-repository ppa:obsproject/obs-studio
sudo apt update
sudo apt install obs-rtspserver
```

#### For Mac:
1. Download the `.pkg` file from the releases page
2. Install the package
3. Restart OBS

### Step 3: Configure OBS for RTSP

1. **Add Video Source in OBS:**
   - Open OBS Studio
   - In the "Sources" panel, click the `+` button
   - Select "Video Capture Device"
   - Name it (e.g., "Webcam")
   - Select your camera from the dropdown
   - Click "OK"

2. **Configure RTSP Server:**
   - Go to `Tools` → `RTSP Server Settings`
   - **Enable RTSP Server:** Check this box
   - **Server Port:** 8554 (default, or choose another)
   - **URL Suffix:** `/live` (this will be part of your RTSP URL)
   - **Auto Start:** Check this to start automatically
   - **Enable Authentication:** (Optional) Add username/password for security
   
   Your RTSP URL will be: `rtsp://127.0.0.1:8554/live`
   
3. **Click "OK"** to save settings

4. **Start Streaming:**
   - Click "Start Streaming" in OBS
   - The RTSP server should now be broadcasting

### Step 4: Configure Your Python Application

Open `config.py` and update these settings:

```python
# Enable RTSP mode
USE_RTSP = True

# Set your RTSP URL
RTSP_URL = "rtsp://127.0.0.1:8554/live"

# If you enabled authentication in OBS:
# RTSP_URL = "rtsp://username:password@127.0.0.1:8554/live"
```

### Step 5: Test the Connection

1. Keep OBS running with streaming started
2. Run your face detection system:
   ```bash
   python main.py
   ```
3. Select option 1 to start detection
4. You should see the RTSP stream connected successfully!

---

## Method 2: MediaMTX RTSP Server (Alternative)

MediaMTX is a lightweight RTSP server that's easier to set up for camera streaming.

### Step 1: Download MediaMTX

**Windows:**
1. Download from: https://github.com/bluenviron/mediamtx/releases
2. Download `mediamtx_vX.X.X_windows_amd64.zip`
3. Extract to a folder (e.g., `C:\mediamtx`)

**Linux:**
```bash
wget https://github.com/bluenviron/mediamtx/releases/latest/download/mediamtx_linux_amd64.tar.gz
tar -xzf mediamtx_linux_amd64.tar.gz
```

### Step 2: Configure MediaMTX

1. Edit `mediamtx.yml`:
```yaml
paths:
  cam:
    source: "v4l2:///dev/video0"  # Linux
    # source: "dshow://video={Your Camera Name}"  # Windows
```

2. Run MediaMTX:
```bash
# Windows
mediamtx.exe

# Linux
./mediamtx
```

3. Your camera will be available at: `rtsp://localhost:8554/cam`

---

## Method 3: VLC RTSP Streaming (Quick Test)

### For Quick Testing:

1. **Open VLC Media Player**
2. Go to `Media` → `Stream`
3. Click `Add` and select your camera (under "Capture Device")
4. Click `Stream` button
5. Click `Next`
6. Select `RTP / MPEG Transport Stream`
7. Click `Add`
8. Enter:
   - Address: `127.0.0.1`
   - Port: `8554`
   - Stream name: `/live`
9. Click `Next` and then `Stream`

Your stream URL: `rtsp://127.0.0.1:8554/live`

---

## Common Issues and Solutions

### Issue 1: "Could not connect to RTSP stream"
**Solutions:**
- Ensure OBS is running and streaming is started
- Check if the RTSP server plugin is installed correctly
- Verify the RTSP URL matches in both OBS and config.py
- Check firewall settings (allow port 8554)
- Try using `127.0.0.1` instead of `localhost`

### Issue 2: High Latency
**Solutions:**
- In `config.py`, reduce `RTSP_BUFFER_SIZE` to 1
- Use TCP transport: `RTSP_TRANSPORT = "tcp"`
- In OBS, reduce video resolution
- Use a wired network connection instead of WiFi

### Issue 3: Stream Keeps Disconnecting
**Solutions:**
- Increase `RTSP_RECONNECT_ATTEMPTS` in config.py
- Check network stability
- Reduce video quality in OBS
- Use TCP instead of UDP transport

### Issue 4: Black Screen or Frozen Frame
**Solutions:**
- Restart OBS and start streaming again
- Check if camera is working in other applications
- Update camera drivers
- Try reducing resolution in OBS

---

## Testing Your RTSP Stream

Before using with the face detection system, test your RTSP stream with VLC:

1. Open VLC Media Player
2. Go to `Media` → `Open Network Stream`
3. Enter your RTSP URL: `rtsp://127.0.0.1:8554/live`
4. Click `Play`

If you see your camera feed in VLC, the RTSP stream is working correctly!

---

## Network Camera (IP Camera) Setup

If you have an IP camera with RTSP support:

1. Find your camera's RTSP URL (check camera documentation)
   - Common format: `rtsp://username:password@192.168.1.100:554/stream`
   
2. Update `config.py`:
```python
USE_RTSP = True
RTSP_URL = "rtsp://admin:password@192.168.1.100:554/stream1"
```

3. Common IP Camera RTSP URLs:
   - **Hikvision:** `rtsp://user:pass@ip:554/Streaming/Channels/101`
   - **Dahua:** `rtsp://user:pass@ip:554/cam/realmonitor?channel=1&subtype=0`
   - **Reolink:** `rtsp://user:pass@ip:554/h264Preview_01_main`
   - **TP-Link:** `rtsp://user:pass@ip:554/stream1`

---

## Performance Tips

1. **Lower Resolution:** In OBS output settings, use 640x480 for faster processing
2. **Reduce Frame Rate:** Set to 15-30 FPS in OBS
3. **Use H.264 Codec:** More efficient than other codecs
4. **Local Network:** Keep RTSP server on the same machine or local network
5. **Wired Connection:** Use Ethernet instead of WiFi for stability

---

## Quick Start Commands

### To enable RTSP in your application:

```bash
# Open config.py and change:
USE_RTSP = True
RTSP_URL = "rtsp://127.0.0.1:8554/live"

# Then run:
python main.py
```

### To switch back to webcam:

```bash
# Open config.py and change:
USE_RTSP = False

# Then run:
python main.py
```

---

## Additional Resources

- **OBS Studio:** https://obsproject.com
- **OBS RTSP Plugin:** https://github.com/iamscottxu/obs-rtspserver
- **MediaMTX:** https://github.com/bluenviron/mediamtx
- **VLC Media Player:** https://www.videolan.org/vlc/

---

## Support

If you encounter issues:
1. Check the error messages in the console
2. Test RTSP stream with VLC player first
3. Verify firewall settings
4. Ensure OBS is streaming
5. Check the RTSP URL matches exactly

For IP camera issues:
- Consult your camera's manual for the correct RTSP URL format
- Test the URL in VLC first
- Verify network connectivity to the camera
