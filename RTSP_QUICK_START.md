# RTSP Quick Start Guide

## 🚀 Quick Setup (5 Minutes)

### Phone Camera Input (Fastest Setup)

1. Connect phone and laptop to the same Wi-Fi.
2. Start your phone camera stream app.
   - Android IP Webcam example RTSP: `rtsp://PHONE_IP:8080/h264.sdp`
3. Run:
   ```bash
   setup_phone_camera.bat
   ```
4. Start relay:
   ```bash
   start_go2rtc.bat
   ```
5. Run app:
   ```bash
   python main.py
   ```

The app will use the local relay URL `rtsp://127.0.0.1:8554/cam`.

---

### Option 0: Using go2rtc Local Relay (Best for Low Latency)

1. **Download go2rtc for Windows**
   - https://github.com/AlexxIT/go2rtc/releases
   - Place `go2rtc.exe` inside this project folder.

2. **Use bundled go2rtc config**
   - File: `go2rtc.yaml`
   - Default stream name is `cam`.
   - Update source URL inside `go2rtc.yaml` if your camera IP changes.

3. **Start relay**
   ```bash
   start_go2rtc.bat
   ```

4. **Verify relay endpoints**
   - RTSP relay: `rtsp://127.0.0.1:8554/cam`
   - go2rtc UI: `http://127.0.0.1:1984`

5. **Run your app**
   - `config.py` is already set to read from go2rtc local relay.
   ```bash
   python main.py
   ```

---

### Option A: Using OBS Studio (Recommended)

1. **Install OBS Studio**
   - Download: https://obsproject.com/download
   - Install and launch OBS

2. **Install RTSP Server Plugin**
   - Download: https://github.com/iamscottxu/obs-rtspserver/releases
   - Run the installer
   - Restart OBS

3. **Configure OBS**
   - Add your camera: Sources → + → Video Capture Device
   - Tools → RTSP Server Settings
     - ✅ Enable RTSP Server
     - Port: `8554`
     - URL Suffix: `/live`
     - Click OK
   - Click "Start Streaming"

4. **Configure Your App**
   - Edit `config.py`:
     ```python
     USE_RTSP = True
     RTSP_URL = "rtsp://127.0.0.1:8554/live"
     ```

5. **Run Your App**
   ```bash
   python main.py
   ```

---

## 🧪 Test Your Setup

### Test RTSP Connection First:
```bash
python test_rtsp.py
```

### Test in VLC Player:
1. Open VLC
2. Media → Open Network Stream
3. URL: `rtsp://127.0.0.1:8554/live`
4. Play

---

## 🔧 Common RTSP URLs

### Local OBS Stream:
```
rtsp://127.0.0.1:8554/live
```

### IP Camera (with authentication):
```
rtsp://username:password@192.168.1.100:554/stream
```

### Common IP Camera URLs:
- **Hikvision:** `rtsp://admin:password@192.168.1.64:554/Streaming/Channels/101`
- **Dahua:** `rtsp://admin:password@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0`
- **Reolink:** `rtsp://admin:password@192.168.1.150:554/h264Preview_01_main`

---

## 📝 Configuration Options (config.py)

```python
# Switch between webcam and RTSP
USE_RTSP = False  # False = Webcam, True = RTSP

# RTSP settings
RTSP_URL = "rtsp://127.0.0.1:8554/live"
RTSP_RECONNECT_ATTEMPTS = 5  # Retry attempts
RTSP_RECONNECT_DELAY = 2     # Seconds between retries
RTSP_BUFFER_SIZE = 1         # Lower = less latency
RTSP_TRANSPORT = "tcp"       # "tcp" or "udp"
```

---

## ⚡ Performance Tips

1. **Reduce Latency:**
   - Set `RTSP_BUFFER_SIZE = 1`
   - Use TCP transport
   - Lower resolution in OBS (640x480)

2. **Improve Stability:**
   - Use wired network (not WiFi)
   - Increase `RTSP_RECONNECT_ATTEMPTS`
   - Check firewall settings

3. **Better Quality:**
   - Increase resolution in OBS
   - Use higher bitrate
   - Ensure good network bandwidth

---

## 🐛 Troubleshooting

### "Could not connect to RTSP stream"
- ✅ Is OBS running and streaming?
- ✅ Is RTSP Server plugin installed?
- ✅ Check RTSP URL matches
- ✅ Firewall allows port 8554?

### High Latency
- Reduce buffer size
- Use TCP transport
- Lower resolution in OBS
- Use wired connection

### Stream Disconnects
- Increase reconnect attempts
- Stabilize network
- Use TCP instead of UDP

---

## 📚 Full Documentation

See [RTSP_SETUP_GUIDE.md](RTSP_SETUP_GUIDE.md) for complete instructions.

---

## 🎯 Quick Commands

### Switch to RTSP:
```bash
# Edit config.py
USE_RTSP = True
RTSP_URL = "rtsp://127.0.0.1:8554/live"

# Run
python main.py
```

### Switch back to Webcam:
```bash
# Edit config.py
USE_RTSP = False

# Run
python main.py
```

---

## ✅ Checklist

Before running the application:

- [ ] OBS installed
- [ ] RTSP Server plugin installed
- [ ] Camera added in OBS
- [ ] RTSP Server enabled in OBS
- [ ] OBS streaming started
- [ ] config.py updated with USE_RTSP = True
- [ ] RTSP URL is correct
- [ ] Tested with test_rtsp.py or VLC

---

**Need Help?** Check the full guide: [RTSP_SETUP_GUIDE.md](RTSP_SETUP_GUIDE.md)
