import cv2
import mediapipe as mp
import numpy as np

# ---- Angle Calculator ----
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180:
        angle = 360 - angle
    return angle

# ---- Settings ----
VIDEO_PATH = "squat.mp4"  # Put your video file name here

mp_pose = mp.solutions.pose

print("\n Loading video...")

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"ERROR: Could not open '{VIDEO_PATH}'")
    print("Make sure the video is in the same folder as this script.")
    exit()

# Data storage
left_knee_angles  = []
right_knee_angles = []
left_hip_angles   = []
right_hip_angles  = []

frame_count = 0
valid_frames = 0

print("Analyzing squat... please wait.\n")

with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as pose:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:
            valid_frames += 1
            lm = results.pose_landmarks.landmark

            # Left side
            l_shoulder = [lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                          lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            l_hip      = [lm[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                          lm[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            l_knee     = [lm[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                          lm[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            l_ankle    = [lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                          lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

            # Right side
            r_shoulder = [lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                          lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            r_hip      = [lm[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                          lm[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            r_knee     = [lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                          lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            r_ankle    = [lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                          lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

            # Calculate angles
            left_knee_angles.append(calculate_angle(l_hip, l_knee, l_ankle))
            right_knee_angles.append(calculate_angle(r_hip, r_knee, r_ankle))
            left_hip_angles.append(calculate_angle(l_shoulder, l_hip, l_knee))
            right_hip_angles.append(calculate_angle(r_shoulder, r_hip, r_knee))

cap.release()

# ---- Check if anything was detected ----
if valid_frames == 0:
    print("ERROR: No body detected in video.")
    print("Make sure your full body is visible and well lit.")
    exit()

print(f"Analyzed {frame_count} total frames")
print(f"Body detected in {valid_frames} frames\n")

# ---- Calculate Summary ----
avg_knee = (np.mean(left_knee_angles) + np.mean(right_knee_angles)) / 2
min_knee = min(np.min(left_knee_angles), np.min(right_knee_angles))
avg_hip  = (np.mean(left_hip_angles) + np.mean(right_hip_angles)) / 2

# ---- Print Report ----
print("=" * 50)
print("     SQUAT BIOMECHANICAL ANALYSIS REPORT")
print("=" * 50)

print(f"\n MEASUREMENTS:")
print(f"   Average Knee Angle  : {avg_knee:.1f} degrees")
print(f"   Deepest Knee Angle  : {min_knee:.1f} degrees")
print(f"   Average Hip Angle   : {avg_hip:.1f} degrees")

print(f"\n FEEDBACK:\n")

# Depth feedback
if min_knee <= 90:
    print("   DEPTH     : EXCELLENT")
    print("   Full depth achieved. Great range of motion.")
elif min_knee <= 110:
    print("   DEPTH     : AVERAGE")
    print("   Getting close to parallel but not quite.")
    print("   Work on ankle and hip mobility to go deeper.")
else:
    print("   DEPTH     : POOR")
    print("   You are not reaching parallel depth.")
    print("   This reduces muscle activation significantly.")
    print("   Focus on ankle mobility and hip flexor stretching.")

print()

# Posture feedback
if avg_hip >= 60:
    print("   POSTURE   : GOOD")
    print("   Torso is upright. Spine position looks healthy.")
elif avg_hip >= 40:
    print("   POSTURE   : AVERAGE")
    print("   Slight forward lean detected.")
    print("   Brace your core harder before descending.")
else:
    print("   POSTURE   : POOR")
    print("   Excessive forward lean throughout the squat.")
    print("   This puts dangerous load on your lower back.")
    print("   Keep chest up, core tight, look slightly upward.")

print()

# Overall
print("   OVERALL SCORE:")
if min_knee <= 90 and avg_hip >= 60:
    print("   EXCELLENT FORM - Keep training this way.")
elif min_knee <= 110 or avg_hip >= 40:
    print("   AVERAGE FORM - Small corrections will make big difference.")
else:
    print("   NEEDS WORK - Focus on mobility and form before adding weight.")

print("\n" + "=" * 50 + "\n")