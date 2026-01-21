# Multi-Angle Face Registration - Visual Guide

## Registration Flow

```
┌─────────────────────────────────────────────┐
│   STEP 1: Enter Student Details            │
├─────────────────────────────────────────────┤
│                                             │
│   Student ID:  [S001____________]           │
│                                             │
│   Name:        [John Doe_________]          │
│                                             │
│   Department:  [Computer Science_]          │
│                                             │
│   Year:        [▼ 2            ]            │
│                                             │
│            [Next: Capture Face ➡]           │
│                                             │
└─────────────────────────────────────────────┘
```

## Multi-Angle Capture Interface

```
┌──────────────────────────────────────────────────────────────┐
│  Registering: John Doe (S001)      0/5 angles captured       │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌───────────────────────────────────────────────┐           │
│  │          Camera Feed                          │           │
│  │                                               │           │
│  │         ┌─────────────────┐                   │           │
│  │         │                 │                   │           │
│  │         │   😊 Your Face  │   ← Green box     │           │
│  │         │                 │      when detected│           │
│  │         └─────────────────┘                   │           │
│  │                                               │           │
│  │         640 x 480 pixels                      │           │
│  └───────────────────────────────────────────────┘           │
│                                                               │
├──────────────────────────────────────────────────────────────┤
│                      Instructions                             │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│                         😊                                    │
│                                                               │
│                  Angle 1: Center                              │
│                                                               │
│            Look straight at the camera                        │
│                                                               │
├──────────────────────────────────────────────────────────────┤
│  [📷 Capture This Angle]              [✓ Finish] [Cancel]    │
└──────────────────────────────────────────────────────────────┘
```

## Capture Sequence

### 1️⃣ First Angle - Center (😊)
```
     Camera
        |
        |
       😊  ← Look straight ahead
```
**Instruction**: "Look straight at the camera"

### 2️⃣ Second Angle - Turn Left (😏)
```
     Camera
        |
       /
      😏  ← Turn head left ~15-20°
```
**Instruction**: "Turn your head slightly to the left"

### 3️⃣ Third Angle - Turn Right (😌)
```
     Camera
        |
         \
          😌  ← Turn head right ~15-20°
```
**Instruction**: "Turn your head slightly to the right"

### 4️⃣ Fourth Angle - Look Up (😄)
```
     Camera
        |
        |
       😄  ← Tilt head up ~10-15°
       /\
```
**Instruction**: "Tilt your head slightly up"

### 5️⃣ Fifth Angle - Look Down (🙂)
```
       \/
       🙂  ← Tilt head down ~10-15°
        |
        |
     Camera
```
**Instruction**: "Tilt your head slightly down"

## After All Angles Captured

```
┌──────────────────────────────────────────────────────────────┐
│  Registering: John Doe (S001)      5/5 angles captured  ✓    │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌───────────────────────────────────────────────┐           │
│  │          Camera Feed                          │           │
│  │                                               │           │
│  │               All angles                      │           │
│  │               captured! ✅                     │           │
│  │                                               │           │
│  └───────────────────────────────────────────────┘           │
│                                                               │
├──────────────────────────────────────────────────────────────┤
│                      Instructions                             │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│                         ✅                                    │
│                                                               │
│               All angles captured!                            │
│                                                               │
│        Click 'Finish Registration' to complete               │
│                                                               │
├──────────────────────────────────────────────────────────────┤
│  [📷 Capture This Angle] (disabled)  [✓ Finish] [Cancel]     │
└──────────────────────────────────────────────────────────────┘
```

## Database Storage

After successful registration, the following data is stored:

### Students Table
```
┌────┬────────────┬──────────┬──────────────────┬──────┬────────────┐
│ ID │ Student ID │   Name   │    Department    │ Year │  Created   │
├────┼────────────┼──────────┼──────────────────┼──────┼────────────┤
│ 1  │   S001     │ John Doe │ Computer Science │  2   │ 2026-01-20 │
└────┴────────────┴──────────┴──────────────────┴──────┴────────────┘
```

### Face Images Table (NEW)
```
┌────┬────────────┬───────────────────┬──────────────────┐
│ ID │ Student ID │ Angle Description │  Capture Order   │
├────┼────────────┼───────────────────┼──────────────────┤
│ 1  │   S001     │     Center        │        1         │
│ 2  │   S001     │    Turn Left      │        2         │
│ 3  │   S001     │    Turn Right     │        3         │
│ 4  │   S001     │     Look Up       │        4         │
│ 5  │   S001     │    Look Down      │        5         │
└────┴────────────┴───────────────────┴──────────────────┘
```

### Stored Files
```
database/faces/
  ├── S001_angle1_center_20260120_143022.jpg
  ├── S001_angle2_turn_left_20260120_143035.jpg
  ├── S001_angle3_turn_right_20260120_143048.jpg
  ├── S001_angle4_look_up_20260120_143102.jpg
  └── S001_angle5_look_down_20260120_143115.jpg
```

## Recognition Process

When a student enters the canteen:

```
┌─────────────────────────────────────────────────────────────┐
│                    Recognition Flow                          │
└─────────────────────────────────────────────────────────────┘

   Camera captures face
          ↓
   Extract face embedding
          ↓
   Compare with ALL 5 stored angles for each student
          ↓
   ┌─────────────────────────────────────────────┐
   │  Similarity Scores:                         │
   │  • Center:      87%                         │
   │  • Turn Left:   82%                         │
   │  • Turn Right:  85%                         │
   │  • Look Up:     79%                         │
   │  • Look Down:   81%                         │
   │                                             │
   │  Final Score = 0.7×max + 0.3×avg           │
   │              = 0.7×87% + 0.3×82.8%         │
   │              = 85.7% ✓ RECOGNIZED           │
   └─────────────────────────────────────────────┘
          ↓
   Log visit to database
          ↓
   Display notification
```

## Comparison: Old vs New System

### Old System (Single Image)
```
Registration: 1 photo from center only
Recognition: Compare against 1 embedding
Accuracy: Good for frontal view only
Problem: Fails at angles or different lighting
```

### New System (Multi-Angle)
```
Registration: 5 photos from different angles
Recognition: Compare against 5 embeddings
Accuracy: Excellent from any angle
Benefit: Works with pose variations & lighting changes
```

## Success Message

```
╔════════════════════════════════════════╗
║          🎉 Success! 🎉                ║
╠════════════════════════════════════════╣
║                                        ║
║  Student John Doe registered           ║
║  successfully with 5 face angles!      ║
║                                        ║
║  ✅ Center                             ║
║  ✅ Turn Left                          ║
║  ✅ Turn Right                         ║
║  ✅ Look Up                            ║
║  ✅ Look Down                          ║
║                                        ║
║  The student can now be recognized     ║
║  from multiple angles and lighting     ║
║  conditions!                           ║
║                                        ║
╚════════════════════════════════════════╝
```

---

**Enjoy the improved accuracy!** 🎯
