# Videos Folder

Place video files here for automatic processing by the Video Processing Queue System.

## Supported Formats

- `.mp4`
- `.avi`
- `.mov`
- `.mkv`
- `.flv`

## Video Requirements

- **Maximum Duration**: 10 minutes
- **File Status**: Must not have `_done` suffix in filename

## How It Works

1. Place your video files in this folder
2. Go to the GUI application → **🎥 Video Processing** tab
3. Click **▶ Start Processing**
4. The system will:
   - Auto-detect faces in the video
   - Recognize registered students
   - Log visits with timestamps
   - Move processed videos to `processed_videos/` folder

## Alternative: Manual Upload

You can also use the **📁 Upload Video** button in the GUI to select and add videos to this folder.

## Processing Details

- Frame Sampling: 1 frame every 10 frames (for faster processing)
- Recognition logs include video timestamp (HH:MM:SS)
- Duplicate recognitions within 30 seconds are filtered
- All results are saved to the database

## Folder Structure After Processing

```
videos/               # Incoming videos (this folder)
processed_videos/     # Completed videos (renamed with _done suffix)
logs/                 # Processing logs
```

## Notes

- Files with `_done` suffix are automatically skipped
- Hidden files (starting with `.` or `~`) are ignored
- Processing runs in background thread (non-blocking)
- System waits 60 seconds when folder is empty, then rescans
