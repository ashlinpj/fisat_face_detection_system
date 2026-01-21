# Multi-Angle Face Recognition Setup

## Overview

The system now supports **multi-angle face capture** similar to smartphone face recognition systems like iPhone Face ID or Android Face Unlock. This significantly improves recognition accuracy by capturing faces from different angles and lighting conditions.

## Features

### 🎯 Enhanced Accuracy
- **5 Different Angles**: Center, Left, Right, Up, and Down
- **Better Recognition**: Works from different viewing angles
- **Lighting Variations**: Captures face in various lighting conditions
- **Higher Confidence**: Multiple embeddings provide more robust matching

### 📱 Smartphone-Like Experience
- **Guided Capture**: Step-by-step instructions with visual feedback
- **Emoji Indicators**: Clear visual cues for each angle (😊😏😌😄🙂)
- **Real-time Preview**: Live camera feed with face detection overlay
- **Progress Tracking**: Shows how many angles have been captured

## How to Register a New Student

### Step 1: Start Detection
1. Open the application
2. Go to the **"📹 Live Detection"** tab
3. Click **"▶ Start Logging"** to activate the camera

### Step 2: Begin Registration
1. Click **"➕ Register New Student"** button
2. Fill in student details:
   - Student ID (required)
   - Name (required)
   - Department (optional)
   - Year (1-4)
3. Click **"Next: Capture Face ➡"**

### Step 3: Multi-Angle Capture
The system will guide you through capturing 5 different angles:

#### Angle 1: Center (😊)
- **Instruction**: "Look straight at the camera"
- Position yourself directly facing the camera
- Click **"📷 Capture This Angle"**

#### Angle 2: Turn Left (😏)
- **Instruction**: "Turn your head slightly to the left"
- Turn your head about 15-20 degrees to your left
- Keep your face visible to the camera
- Click **"📷 Capture This Angle"**

#### Angle 3: Turn Right (😌)
- **Instruction**: "Turn your head slightly to the right"
- Turn your head about 15-20 degrees to your right
- Keep your face visible to the camera
- Click **"📷 Capture This Angle"**

#### Angle 4: Look Up (😄)
- **Instruction**: "Tilt your head slightly up"
- Tilt your head upward about 10-15 degrees
- Keep looking at the camera
- Click **"📷 Capture This Angle"**

#### Angle 5: Look Down (🙂)
- **Instruction**: "Tilt your head slightly down"
- Tilt your head downward about 10-15 degrees
- Keep your face visible
- Click **"📷 Capture This Angle"**

### Step 4: Complete Registration
- After all 5 angles are captured, the progress will show "5/5 angles captured"
- Click **"✓ Finish Registration"** to save the student profile
- All face images and embeddings are stored in the database

## Technical Details

### Database Structure

#### Students Table
Stores basic student information with primary face embedding:
```sql
CREATE TABLE students (
    id INTEGER PRIMARY KEY,
    student_id TEXT UNIQUE,
    name TEXT,
    department TEXT,
    year INTEGER,
    face_embedding TEXT,
    face_image_path TEXT,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
)
```

#### Face Images Table (NEW)
Stores multiple face angles per student:
```sql
CREATE TABLE face_images (
    id INTEGER PRIMARY KEY,
    student_db_id INTEGER,
    student_id TEXT,
    face_embedding TEXT,
    face_image_path TEXT,
    angle_description TEXT,
    capture_order INTEGER,
    created_at TIMESTAMP,
    FOREIGN KEY (student_db_id) REFERENCES students(id)
)
```

### Recognition Algorithm

The system uses an improved recognition algorithm:

1. **Multi-Embedding Comparison**: For each person, compare against all stored angles
2. **Best + Average Scoring**: Uses 70% best match + 30% average of all angles
3. **Confidence Thresholding**: Requires high confidence to prevent false positives
4. **Gap Analysis**: Ensures significant difference between top 2 matches

```python
# Weighted similarity calculation
final_similarity = 0.7 * max_similarity + 0.3 * avg_similarity
```

### File Storage

Face images are stored in: `database/faces/`

Naming convention:
```
{student_id}_angle{number}_{angle_name}_{timestamp}.jpg

Examples:
- S001_angle1_center_20260120_143022.jpg
- S001_angle2_turn_left_20260120_143035.jpg
- S001_angle3_turn_right_20260120_143048.jpg
```

## Configuration

You can customize the capture angles in `config.py`:

```python
# Number of angles to capture
NUM_CAPTURE_ANGLES = 5

# Define custom angles
CAPTURE_ANGLES = [
    {"name": "Center", "description": "Look straight at the camera", "emoji": "😊"},
    {"name": "Turn Left", "description": "Turn your head slightly to the left", "emoji": "😏"},
    {"name": "Turn Right", "description": "Turn your head slightly to the right", "emoji": "😌"},
    {"name": "Look Up", "description": "Tilt your head slightly up", "emoji": "😄"},
    {"name": "Look Down", "description": "Tilt your head slightly down", "emoji": "🙂"}
]
```

## Benefits

### For Users
- ✅ **Faster Recognition**: No need to position perfectly
- ✅ **Works at Angles**: Recognized from different viewing positions
- ✅ **Better Lighting**: Works in various lighting conditions
- ✅ **Higher Accuracy**: Reduced false positives and false negatives

### For System
- ✅ **Robust Matching**: Multiple reference points per person
- ✅ **Confidence Scoring**: Better discrimination between similar faces
- ✅ **Reduced Errors**: Lower chance of misidentification
- ✅ **Scalability**: Handles larger databases with less confusion

## Troubleshooting

### Problem: Face Not Detected
**Solution**: 
- Ensure adequate lighting
- Move closer to the camera
- Remove obstructions (glasses, masks, hats)

### Problem: Capture Button Not Working
**Solution**:
- Check if camera is active (green detection box should be visible)
- Restart the application if camera feed freezes

### Problem: Low Recognition Accuracy
**Solution**:
- Re-register the student with better lighting
- Ensure all 5 angles are captured clearly
- Check camera quality and positioning

## Migration from Old System

If you have students registered with the old single-image system:
- They will continue to work with legacy support
- Re-register them to get multi-angle benefits
- Old images are preserved in the database

## Future Enhancements

Potential improvements:
- 📸 Add more capture angles (7-9 angles)
- 🌙 Low-light mode with adaptive brightness
- 📹 Video-based registration (capture from video stream)
- 🔄 Auto-angle detection (system suggests optimal angles)
- 📊 Quality scoring for each captured angle

## Support

For issues or questions:
1. Check error messages in the application
2. Review console output for debugging information
3. Verify database integrity with `database/canteen.db`

---

**Version**: 2.0 with Multi-Angle Support  
**Last Updated**: January 20, 2026
