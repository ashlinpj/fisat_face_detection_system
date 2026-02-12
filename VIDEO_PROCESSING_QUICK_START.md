# 🎥 Video Processing - Quick Start Guide

Get started with video processing in 3 minutes!

---

## ⚡ Quick Start (3 Steps)

### Step 1: Launch GUI

```bash
python gui_app.py
```

### Step 2: Upload Video

1. Click on **🎥 Video Processing** tab
2. Click **📁 Upload Video** button
3. Select a video file (MP4, AVI, MOV)

### Step 3: Start Processing

1. Click **▶ Start Processing**
2. Watch recognition results appear in real-time!

---

## 📹 Supported Video Formats

✅ `.mp4`  
✅ `.avi`  
✅ `.mov`  
✅ `.mkv`  
✅ `.flv`

**Max Duration**: 10 minutes

---

## 🎯 What Happens?

1. **System scans** the video frame by frame
2. **Detects faces** using YOLOv8
3. **Recognizes students** using DeepFace
4. **Logs results** with timestamp (HH:MM:SS)
5. **Saves to database** with video name
6. **Moves video** to `processed_videos/` folder

---

## 📊 Where Are Results?

### In GUI

- **Video Processing Tab**: Real-time recognition results
- **Visit Logs Tab**: All database entries
- **Statistics Tab**: Updated statistics

### In Files

- **Database**: `database/canteen.db` → `visit_logs` table
- **Log File**: `logs/video_processing.log`
- **Processed Videos**: `processed_videos/` folder

---

## 💡 Example Workflow

```
1. You have: cafeteria_recording.mp4 (5 minutes)

2. Place in: videos/cafeteria_recording.mp4

3. Start processing in GUI

4. System finds these faces:
   - 00:01:23 - John Doe (95%)
   - 00:02:45 - Jane Smith (92%)
   - 00:04:12 - John Doe (94%)

5. Results saved to database with:
   - Source: "video"
   - Video Name: "cafeteria_recording.mp4"
   - Timestamps: "00:01:23", "00:02:45", "00:04:12"

6. Video moved to: processed_videos/cafeteria_recording_done.mp4
```

---

## 🔧 Need to Change Settings?

Edit `config.py`:

```python
# Process more frames (slower but more accurate)
VIDEO_FRAME_SKIP_INTERVAL = 5

# Process fewer frames (faster but might miss some)
VIDEO_FRAME_SKIP_INTERVAL = 20

# Allow longer videos
VIDEO_MAX_DURATION = 30 * 60  # 30 minutes
```

---

## ❓ Troubleshooting

### Video not processing?

- Check format is supported
- Ensure duration < 10 minutes
- Verify filename doesn't have `_done`

### No faces recognized?

- Check students are registered first
- Ensure video has clear face visibility
- Try lowering `FACE_RECOGNITION_THRESHOLD` in config.py

### Processing too slow?

- Increase `VIDEO_FRAME_SKIP_INTERVAL` (process fewer frames)
- Use shorter videos
- Close other applications

---

## 📁 Folder Structure

```
videos/               → Place videos here
processed_videos/     → Completed videos appear here
logs/                 → Processing logs saved here
```

---

## 🎓 Pro Tips

1. **Test with short video first** (1-2 minutes)
2. **Use good quality videos** for best results
3. **Register students before processing** their videos
4. **Check logs if unsure** what happened
5. **Batch upload** multiple videos overnight

---

## 🚀 Advanced: Batch Processing

Want to process multiple videos automatically?

1. Place all videos in `videos/` folder:

   ```
   videos/
   ├── video1.mp4
   ├── video2.mp4
   └── video3.mp4
   ```

2. Click **▶ Start Processing**

3. System processes them one by one automatically!

4. When done, all videos are in `processed_videos/`

---

## 📞 Need Help?

1. Check **README.md** for full documentation
2. Review **VIDEO_PROCESSING_IMPLEMENTATION.md** for details
3. Check `logs/video_processing.log` for errors
4. Contact your project guide

---

**Happy Processing! 🎉**
