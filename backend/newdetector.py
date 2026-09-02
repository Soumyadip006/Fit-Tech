import cv2
import mediapipe as mp
import numpy as np
import os
from scipy.signal import savgol_filter

# ============================================================
#  FITTECH — BIOMECHANICAL ANALYZER v5.0
#  11 exercises | scipy smoothing | confusion feedback
# ============================================================

mp_pose = mp.solutions.pose

# ---- Angle Between 3 Points ----
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180:
        angle = 360 - angle
    return angle

# ---- Vertical Angle of a Segment ----
def vertical_angle(a, b):
    a = np.array(a)
    b = np.array(b)
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    angle = np.abs(np.degrees(np.arctan2(dx, dy)))
    return angle

# ---- Scipy Savitzky-Golay Smoothing ----
# Research standard for biomechanical signal smoothing
# Preserves peaks and valleys better than moving average
def smooth_series(series, window=9, poly=3):
    arr = np.array(series, dtype=float)
    n = len(arr)
    if n < 5:
        return arr
    w = min(window, n if n % 2 == 1 else n - 1)
    if w < 5:
        return arr
    return savgol_filter(arr, w, min(poly, w - 1))

# ============================================================
#  SMART FILE FINDER
# ============================================================

def _normalise_filename(value):
    stem = os.path.splitext(os.path.basename(value))[0].lower()
    return "".join(ch for ch in stem if ch.isalnum())


def find_video(base_names):
    extensions = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    base_dir = os.path.dirname(os.path.abspath(__file__))
    wanted = {_normalise_filename(name) for name in base_names}
    for root, _, files in os.walk(base_dir):
        for filename in files:
            if os.path.splitext(filename)[1].lower() in extensions:
                if _normalise_filename(filename) in wanted:
                    return os.path.join(root, filename)
    return None


VIDEO_ALIASES = {
    "squat"         : ["squat", "squats", "squat_video", "Squat"],
    "deadlift"      : ["deadlift", "deadlifts", "dead_lift", "Deadlift"],
    "lunge"         : ["lunge", "lunges", "Lunge"],
    "push up"       : ["pushup", "push_up", "push-up", "pushups", "PushUp", "Pushup"],
    "plank"         : ["plank", "planks", "Plank"],
    "pull up"       : ["pullup", "pull_up", "pull-up", "pullups", "PullUp", "Pullup"],
    "sit up"        : ["situp", "sit_up", "sit-up", "situps", "SitUp", "Situp"],
    "leg raise"     : ["legraise", "leg_raise", "leg-raise", "legraises", "LegRaise"],
    "dead hang"     : ["deadhang", "dead_hang", "dead-hang", "DeadHang"],
    "bent over row" : ["bentoverrow", "bent_over_row", "bentover_row", "BentOverRow",
                       "row", "rows"],
    "overhead press": ["overheadpress", "overhead_press", "ohp", "OHP", "OverheadPress",
                       "shoulderpress", "shoulder_press"],
}

VIDEO_ALIASES.update({
    "squat": ["squat", "squats"],
    "deadlift": ["deadlift", "dead_lift", "deadlifts"],
    "lunge": ["lunge", "lunges"],
    "push up": ["pushup", "push_up", "push-up", "pushups"],
    "plank": ["plank", "planks"],
    "pull up": ["pullup", "pull_up", "pull-up", "pullups"],
    "sit up": ["situp", "sit_up", "sit-up", "situps"],
    "leg raise": ["legraise", "leg_raise", "leg-raise", "legraises"],
    "dead hang": ["deadhang", "dead_hang", "dead-hang"],
    "bent over row": ["bentoverrow", "bent_over_row", "bentover_row", "row", "rows"],
    "overhead press": ["overheadpress", "overhead_press", "ohp", "shoulderpress", "shoulder_press"],
})


# ============================================================
#  CONFUSION NOTES
#  Printed when two exercises score within 22 points of each other
# ============================================================

CONFUSION_NOTES = {
    frozenset({"SQUAT", "DEADLIFT"}): {
        "body_part": "Knee angle depth and hip hinge ratio",
        "reason"   : "Your knee bent more than a conventional deadlift allows. Research shows deadlift uses only about 35 degrees of knee flexion — knee angle stays around 145 degrees. A squat goes to 90 degrees or below. Extra knee bend in your deadlift means hips are dropping too low and the movement is becoming squat-like.",
        "fix"      : "For deadlift: Set up with hips ABOVE knee level. Keep shins near vertical. Think of pushing the floor away rather than pulling the bar up.",
    },
    frozenset({"SQUAT", "LUNGE"}): {
        "body_part": "Knee symmetry between left and right leg",
        "reason"   : "One knee is bending noticeably more than the other. In a proper squat both legs should work almost identically. The asymmetry is making the pattern look like a lunge.",
        "fix"      : "For squat: Check stance width and foot position. Add single-leg work like Bulgarian split squats to balance both sides.",
    },
    frozenset({"DEADLIFT", "BENT OVER ROW"}): {
        "body_part": "Elbow movement — this is the primary separator",
        "reason"   : "Both involve a forward-hinged torso. In a deadlift arms hang completely straight with nearly zero elbow movement. The torso also cycles from hinged to fully upright every rep. In a bent over row, elbows actively pull back hard each rep, and the torso stays permanently hinged throughout the entire set without standing up.",
        "fix"      : "For deadlift: Let arms hang dead straight throughout. No pulling motion at all. Stand fully upright at the top of every single rep. For bent over row: Active elbow drive every rep. Stay hinged the entire set — never stand up between reps.",
    },
    frozenset({"PUSH UP", "PLANK"}): {
        "body_part": "Elbow range of motion",
        "reason"   : "Both have a horizontal body position. The only difference is whether the elbows move. A push up has large elbow movement every rep. A plank is a completely static hold — elbows do not move at all.",
        "fix"      : "For plank: Elbows locked in position throughout — zero bending. For push up: Full elbow bend every rep, chest nearly touching the floor at the bottom.",
    },
    frozenset({"PULL UP", "DEAD HANG"}): {
        "body_part": "Elbow range of motion while hanging",
        "reason"   : "Both have wrists above shoulders and body hanging from a bar. In a pull up elbows bend significantly to lift the body. In a dead hang arms stay fully extended throughout with minimal elbow movement.",
        "fix"      : "For dead hang: Arms completely straight the entire time — no pulling. For pull up: Drive elbows down and back — elbow bends from full extension to past 90 degrees at the top.",
    },
    frozenset({"PULL UP", "OVERHEAD PRESS"}): {
        "body_part": "Body orientation and direction of effort",
        "reason"   : "Both involve arms overhead. In a pull up the body hangs and arms pull the body upward. In an overhead press the body stands upright and arms push weight above the head. Completely opposite directions of effort.",
        "fix"      : "For pull up: You should feel your back and biceps pulling your body toward the bar. For overhead press: You should feel your shoulders and triceps pushing the weight away from you.",
    },
    frozenset({"SIT-UP", "LEG RAISE"}): {
        "body_part": "Which body part moves — torso or legs",
        "reason"   : "Both are performed lying down. In a sit-up the torso curls upward while legs stay fixed. In a leg raise the legs raise upward while the torso stays completely flat and stationary.",
        "fix"      : "For sit-up: Core pulls your chest toward your knees — legs stay planted. For leg raise: Legs drive upward — keep your lower back pressed flat to the floor throughout.",
    },
    frozenset({"DEAD HANG", "OVERHEAD PRESS"}): {
        "body_part": "Elbow movement and whether body is standing or hanging",
        "reason"   : "Both have wrists above or near shoulder height. In a dead hang arms are fully extended and body just hangs with no elbow movement. In an overhead press elbows bend at the bottom and extend fully at the top — large elbow range of motion.",
        "fix"      : "For dead hang: Arms stay completely straight. Just breathe and hold. For overhead press: Full elbow movement every rep from 90 degrees at the bottom to full lockout overhead.",
    },
    frozenset({"PLANK", "LEG RAISE"}): {
        "body_part": "Hip range of motion",
        "reason"   : "Both have a horizontal body position with minimal elbow movement. In a plank everything is completely static including the hips. In a leg raise the hip angle changes significantly as legs rise from the floor.",
        "fix"      : "For plank: Zero movement — hold position with everything locked. For leg raise: Legs drive upward creating a clear hip angle change while the rest of the body stays flat.",
    },
}

# ============================================================
#  MAIN ANALYSIS FUNCTION
# ============================================================

def analyze_video(VIDEO_PATH, requested_exercise=None):
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"\n  ERROR: Could not open '{VIDEO_PATH}'")
        print("  Make sure the video is in the BIOMECH folder.\n")
        return

    data = {
        'left_knee'             : [],
        'right_knee'            : [],
        'left_hip'              : [],
        'right_hip'             : [],
        'left_elbow'            : [],
        'right_elbow'           : [],
        'left_shoulder'         : [],
        'right_shoulder'        : [],
        'left_ankle'            : [],
        'right_ankle'           : [],
        'torso_lean'            : [],
        'torso_vertical'        : [],
        'wrist_above_shoulder'  : [],
        'left_wrist_y'          : [],
        'right_wrist_y'         : [],
        'left_shoulder_y'       : [],
        'right_shoulder_y'      : [],
        'hip_y'                 : [],
        'shoulder_y'            : [],
        'left_knee_y'           : [],
        'right_knee_y'          : [],
        'left_ankle_y'          : [],
        'right_ankle_y'         : [],
        'left_shoulder_abduct'  : [],
        'right_shoulder_abduct' : [],
        'knee_series'           : [],
        'elbow_series'          : [],
        'hip_series'            : [],
    }

    print("\n  Analyzing video... please wait.")

    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark

                def pt(landmark):
                    return [lm[landmark.value].x, lm[landmark.value].y]

                l_shoulder  = pt(mp_pose.PoseLandmark.LEFT_SHOULDER)
                r_shoulder  = pt(mp_pose.PoseLandmark.RIGHT_SHOULDER)
                l_elbow     = pt(mp_pose.PoseLandmark.LEFT_ELBOW)
                r_elbow     = pt(mp_pose.PoseLandmark.RIGHT_ELBOW)
                l_wrist     = pt(mp_pose.PoseLandmark.LEFT_WRIST)
                r_wrist     = pt(mp_pose.PoseLandmark.RIGHT_WRIST)
                l_hip       = pt(mp_pose.PoseLandmark.LEFT_HIP)
                r_hip       = pt(mp_pose.PoseLandmark.RIGHT_HIP)
                l_knee      = pt(mp_pose.PoseLandmark.LEFT_KNEE)
                r_knee      = pt(mp_pose.PoseLandmark.RIGHT_KNEE)
                l_ankle     = pt(mp_pose.PoseLandmark.LEFT_ANKLE)
                r_ankle     = pt(mp_pose.PoseLandmark.RIGHT_ANKLE)
                l_foot      = pt(mp_pose.PoseLandmark.LEFT_FOOT_INDEX)
                r_foot      = pt(mp_pose.PoseLandmark.RIGHT_FOOT_INDEX)

                mid_shoulder = [(l_shoulder[0]+r_shoulder[0])/2,
                                (l_shoulder[1]+r_shoulder[1])/2]
                mid_hip      = [(l_hip[0]+r_hip[0])/2,
                                (l_hip[1]+r_hip[1])/2]

                lk = calculate_angle(l_hip, l_knee, l_ankle)
                rk = calculate_angle(r_hip, r_knee, r_ankle)
                data['left_knee'].append(lk)
                data['right_knee'].append(rk)
                data['knee_series'].append((lk + rk) / 2)

                lh = calculate_angle(l_shoulder, l_hip, l_knee)
                rh = calculate_angle(r_shoulder, r_hip, r_knee)
                data['left_hip'].append(lh)
                data['right_hip'].append(rh)
                data['hip_series'].append((lh + rh) / 2)

                le = calculate_angle(l_shoulder, l_elbow, l_wrist)
                re = calculate_angle(r_shoulder, r_elbow, r_wrist)
                data['left_elbow'].append(le)
                data['right_elbow'].append(re)
                data['elbow_series'].append((le + re) / 2)

                ls = calculate_angle(l_hip, l_shoulder, l_elbow)
                rs = calculate_angle(r_hip, r_shoulder, r_elbow)
                data['left_shoulder'].append(ls)
                data['right_shoulder'].append(rs)

                la = calculate_angle(l_knee, l_ankle, l_foot)
                ra = calculate_angle(r_knee, r_ankle, r_foot)
                data['left_ankle'].append(la)
                data['right_ankle'].append(ra)

                lsa = calculate_angle(l_hip, l_shoulder, l_wrist)
                rsa = calculate_angle(r_hip, r_shoulder, r_wrist)
                data['left_shoulder_abduct'].append(lsa)
                data['right_shoulder_abduct'].append(rsa)

                data['torso_lean'].append(vertical_angle(mid_shoulder, mid_hip))
                data['torso_vertical'].append(mid_hip[1] - mid_shoulder[1])

                avg_wrist_y    = (l_wrist[1] + r_wrist[1]) / 2
                avg_shoulder_y = (l_shoulder[1] + r_shoulder[1]) / 2
                data['wrist_above_shoulder'].append(
                    1 if avg_wrist_y < avg_shoulder_y else 0)

                data['left_wrist_y'].append(l_wrist[1])
                data['right_wrist_y'].append(r_wrist[1])
                data['left_shoulder_y'].append(l_shoulder[1])
                data['right_shoulder_y'].append(r_shoulder[1])
                data['hip_y'].append(mid_hip[1])
                data['shoulder_y'].append(mid_shoulder[1])
                data['left_knee_y'].append(l_knee[1])
                data['right_knee_y'].append(r_knee[1])
                data['left_ankle_y'].append(l_ankle[1])
                data['right_ankle_y'].append(r_ankle[1])

    cap.release()

    if len(data['left_knee']) == 0:
        print("  ERROR: No body detected in video.")
        print("  Tips: Record from side angle. Full body visible. Good lighting.\n")
        return

    total_frames = len(data['left_knee'])

    # ---- Smooth with scipy Savitzky-Golay filter ----
    smooth_keys = [
        'left_knee','right_knee','left_hip','right_hip',
        'left_elbow','right_elbow','left_shoulder','right_shoulder',
        'left_ankle','right_ankle','torso_lean','torso_vertical',
        'left_shoulder_abduct','right_shoulder_abduct',
        'knee_series','hip_series','elbow_series'
    ]
    for key in smooth_keys:
        data[key] = smooth_series(data[key])

    # ============================================================
    #  COMPUTE ALL STATS
    # ============================================================

    avg_left_knee    = np.mean(data['left_knee'])
    avg_right_knee   = np.mean(data['right_knee'])
    avg_knee         = (avg_left_knee + avg_right_knee) / 2
    min_knee         = min(np.min(data['left_knee']), np.min(data['right_knee']))
    max_knee         = max(np.max(data['left_knee']), np.max(data['right_knee']))
    knee_range       = max_knee - min_knee
    knee_std         = np.std(data['knee_series'])
    knee_asymmetry   = abs(avg_left_knee - avg_right_knee)

    avg_left_hip     = np.mean(data['left_hip'])
    avg_right_hip    = np.mean(data['right_hip'])
    avg_hip          = (avg_left_hip + avg_right_hip) / 2
    min_hip          = min(np.min(data['left_hip']), np.min(data['right_hip']))
    max_hip          = max(np.max(data['left_hip']), np.max(data['right_hip']))
    hip_range        = max_hip - min_hip

    avg_left_elbow   = np.mean(data['left_elbow'])
    avg_right_elbow  = np.mean(data['right_elbow'])
    avg_elbow        = (avg_left_elbow + avg_right_elbow) / 2
    min_elbow        = min(np.min(data['left_elbow']), np.min(data['right_elbow']))
    max_elbow        = max(np.max(data['left_elbow']), np.max(data['right_elbow']))
    elbow_range      = max_elbow - min_elbow

    avg_shoulder     = (np.mean(data['left_shoulder']) +
                        np.mean(data['right_shoulder'])) / 2
    avg_shoulder_abduct = (np.mean(data['left_shoulder_abduct']) +
                           np.mean(data['right_shoulder_abduct'])) / 2
    max_shoulder_abduct = max(np.max(data['left_shoulder_abduct']),
                              np.max(data['right_shoulder_abduct']))

    avg_ankle        = (np.mean(data['left_ankle']) +
                        np.mean(data['right_ankle'])) / 2

    avg_torso_lean   = np.mean(data['torso_lean'])
    avg_torso_vert   = np.mean(data['torso_vertical'])
    torso_vert_range = (np.max(data['torso_vertical']) -
                        np.min(data['torso_vertical']))
    torso_vert_std   = np.std(data['torso_vertical'])

    wrist_above_pct  = np.mean(data['wrist_above_shoulder']) * 100
    avg_wrist_y      = (np.mean(data['left_wrist_y']) +
                        np.mean(data['right_wrist_y'])) / 2
    avg_shoulder_y   = (np.mean(data['left_shoulder_y']) +
                        np.mean(data['right_shoulder_y'])) / 2

    knee_y_asymmetry = abs(np.mean(data['left_knee_y']) -
                           np.mean(data['right_knee_y']))

    hip_to_knee_ratio = hip_range / (knee_range + 1e-6)

    # ============================================================
    #  CONFIDENCE SCORING — All 11 exercises scored 0 to 100+
    # ============================================================

    scores = {
        "SQUAT"         : 0,
        "DEADLIFT"      : 0,
        "LUNGE"         : 0,
        "PUSH UP"       : 0,
        "PLANK"         : 0,
        "PULL UP"       : 0,
        "SIT-UP"        : 0,
        "LEG RAISE"     : 0,
        "DEAD HANG"     : 0,
        "BENT OVER ROW" : 0,
        "OVERHEAD PRESS": 0,
    }

    # ============================================================
    #  SQUAT
    #  Knee dominant. Torso upright. Symmetric. Large knee range.
    # ============================================================
    if knee_range > 60:           scores["SQUAT"] += 25
    if knee_range > 90:           scores["SQUAT"] += 10
    if min_knee < 130:            scores["SQUAT"] += 20
    if min_knee < 100:            scores["SQUAT"] += 15
    if min_knee < 90:             scores["SQUAT"] += 10
    if knee_asymmetry < 20:       scores["SQUAT"] += 15
    if avg_torso_vert > 0.22:     scores["SQUAT"] += 20
    if avg_torso_vert > 0.26:     scores["SQUAT"] += 10
    if knee_std > 15:             scores["SQUAT"] += 10
    if hip_to_knee_ratio < 0.8:   scores["SQUAT"] += 15
    # Penalties
    if avg_torso_vert < 0.15:     scores["SQUAT"] -= 40
    if avg_torso_vert < 0.10:     scores["SQUAT"] -= 30
    if knee_range < 30:           scores["SQUAT"] -= 35
    if elbow_range > 80:          scores["SQUAT"] -= 20
    if avg_torso_lean > 30:       scores["SQUAT"] -= 25
    if wrist_above_pct > 70:      scores["SQUAT"] -= 25

    # ============================================================
    #  DEADLIFT
    #  Hip dominant. Torso cycles upright->forward->upright each rep.
    #  Arms hang STRAIGHT = very small elbow range.
    #  This elbow condition is the #1 separator from bent over row.
    # ============================================================
    if knee_range < 60:           scores["DEADLIFT"] += 20
    if knee_range < 40:           scores["DEADLIFT"] += 15
    if min_knee > 110:            scores["DEADLIFT"] += 15
    if min_knee > 125:            scores["DEADLIFT"] += 10
    if min_hip < 60:              scores["DEADLIFT"] += 25
    if min_hip < 40:              scores["DEADLIFT"] += 20
    if min_hip < 30:              scores["DEADLIFT"] += 10
    if avg_torso_vert < 0.25:     scores["DEADLIFT"] += 15
    if avg_torso_vert < 0.18:     scores["DEADLIFT"] += 15
    if avg_torso_lean > 20:       scores["DEADLIFT"] += 20
    if avg_torso_lean > 35:       scores["DEADLIFT"] += 15
    if hip_range > 40:            scores["DEADLIFT"] += 15
    if hip_range > 60:            scores["DEADLIFT"] += 10
    if hip_to_knee_ratio > 1.2:   scores["DEADLIFT"] += 20
    if hip_to_knee_ratio > 1.8:   scores["DEADLIFT"] += 10
    if torso_vert_range > 0.12:   scores["DEADLIFT"] += 20
    if torso_vert_range > 0.18:   scores["DEADLIFT"] += 15
    if knee_asymmetry < 15:       scores["DEADLIFT"] += 5
    # ELBOW SEPARATOR — arms hang dead straight in deadlift
    if elbow_range < 20:          scores["DEADLIFT"] += 25
    if elbow_range < 12:          scores["DEADLIFT"] += 15
    if elbow_range > 35:          scores["DEADLIFT"] -= 30
    if elbow_range > 55:          scores["DEADLIFT"] -= 30
    if elbow_range > 75:          scores["DEADLIFT"] -= 20
    # Other penalties
    if min_knee < 100:            scores["DEADLIFT"] -= 35
    if min_knee < 90:             scores["DEADLIFT"] -= 20
    if knee_range > 80:           scores["DEADLIFT"] -= 30
    if avg_torso_vert > 0.30:     scores["DEADLIFT"] -= 25
    if torso_vert_std < 0.02:     scores["DEADLIFT"] -= 20
    if hip_to_knee_ratio < 0.7:   scores["DEADLIFT"] -= 20
    if wrist_above_pct > 70:      scores["DEADLIFT"] -= 30

    # ============================================================
    #  LUNGE
    #  Significant left vs right knee asymmetry.
    #  One knee bends deeply, other stays back.
    # ============================================================
    if knee_asymmetry > 20:       scores["LUNGE"] += 25
    if knee_asymmetry > 35:       scores["LUNGE"] += 20
    if knee_y_asymmetry > 0.05:   scores["LUNGE"] += 20
    if min_knee < 130:            scores["LUNGE"] += 15
    if knee_range > 40:           scores["LUNGE"] += 10
    if avg_torso_vert > 0.15:     scores["LUNGE"] += 10
    # Penalties
    if knee_asymmetry < 10:       scores["LUNGE"] -= 35
    if knee_y_asymmetry < 0.02:   scores["LUNGE"] -= 25
    if wrist_above_pct > 70:      scores["LUNGE"] -= 25

    # ============================================================
    #  PUSH UP
    #  Horizontal torso. Large elbow range every rep.
    #  Key separator from plank: elbow_range is LARGE.
    # ============================================================
    if avg_torso_vert < 0.10:     scores["PUSH UP"] += 35
    if avg_torso_vert < 0.06:     scores["PUSH UP"] += 20
    if elbow_range > 50:          scores["PUSH UP"] += 30
    if elbow_range > 70:          scores["PUSH UP"] += 15
    if min_elbow < 100:           scores["PUSH UP"] += 15
    if min_elbow < 80:            scores["PUSH UP"] += 15
    if avg_knee > 150:            scores["PUSH UP"] += 10
    # Penalties
    if avg_torso_vert > 0.15:     scores["PUSH UP"] -= 40
    if knee_range > 30:           scores["PUSH UP"] -= 25
    if elbow_range < 20:          scores["PUSH UP"] -= 35
    if wrist_above_pct > 70:      scores["PUSH UP"] -= 30

    # ============================================================
    #  PLANK
    #  Horizontal torso. STATIC hold. Elbows DO NOT MOVE.
    #  Key separator from push up: elbow_range is TINY.
    #  Key separator from leg raise: hip_range also tiny.
    # ============================================================
    if avg_torso_vert < 0.10:     scores["PLANK"] += 35
    if avg_torso_vert < 0.06:     scores["PLANK"] += 20
    if elbow_range < 15:          scores["PLANK"] += 35
    if elbow_range < 8:           scores["PLANK"] += 20
    if avg_knee > 150:            scores["PLANK"] += 10
    if knee_range < 10:           scores["PLANK"] += 15
    if knee_std < 5:              scores["PLANK"] += 15
    if hip_range < 15:            scores["PLANK"] += 15
    if torso_vert_std < 0.02:     scores["PLANK"] += 20
    # Penalties
    if avg_torso_vert > 0.15:     scores["PLANK"] -= 40
    if elbow_range > 30:          scores["PLANK"] -= 45
    if knee_range > 20:           scores["PLANK"] -= 25
    if wrist_above_pct > 70:      scores["PLANK"] -= 30

    # ============================================================
    #  PULL UP
    #  Hanging from bar. Wrists above shoulders throughout.
    #  Elbows bend significantly to pull body up.
    #  Key separator from dead hang: elbow_range is LARGE.
    # ============================================================
    if wrist_above_pct > 70:      scores["PULL UP"] += 30
    if wrist_above_pct > 85:      scores["PULL UP"] += 20
    if elbow_range > 60:          scores["PULL UP"] += 30
    if elbow_range > 80:          scores["PULL UP"] += 15
    if min_elbow < 90:            scores["PULL UP"] += 20
    if min_elbow < 70:            scores["PULL UP"] += 10
    if avg_torso_vert > 0.10:     scores["PULL UP"] += 10
    # Penalties
    if wrist_above_pct < 50:      scores["PULL UP"] -= 40
    if elbow_range < 30:          scores["PULL UP"] -= 40
    if elbow_range < 15:          scores["PULL UP"] -= 20
    if avg_torso_vert < 0.05:     scores["PULL UP"] -= 20
    if knee_range > 60:           scores["PULL UP"] -= 30

    # ============================================================
    #  SIT-UP
    #  Person on floor. Torso curls from horizontal to upright.
    #  Torso_vert_range HIGH. Knees bent ~90 degrees.
    #  Elbow barely moves. Key separator from leg raise: TORSO moves.
    # ============================================================
    if torso_vert_range > 0.15:   scores["SIT-UP"] += 30
    if torso_vert_range > 0.22:   scores["SIT-UP"] += 20
    if avg_torso_vert < 0.18:     scores["SIT-UP"] += 15
    if hip_range > 25:            scores["SIT-UP"] += 20
    if elbow_range < 30:          scores["SIT-UP"] += 20
    if elbow_range < 15:          scores["SIT-UP"] += 10
    if min_knee < 130:            scores["SIT-UP"] += 15
    if torso_vert_std > 0.04:     scores["SIT-UP"] += 15
    # Penalties
    if avg_torso_vert > 0.28:     scores["SIT-UP"] -= 30
    if elbow_range > 60:          scores["SIT-UP"] -= 35
    if wrist_above_pct > 60:      scores["SIT-UP"] -= 30
    if knee_range > 50:           scores["SIT-UP"] -= 25

    # ============================================================
    #  LEG RAISE
    #  Person on floor. Legs raise from ground. Torso STATIC flat.
    #  Large hip_range. Elbow static. Torso_vert_std very low.
    #  Key separator from sit-up: TORSO stays still, legs move.
    #  Key separator from plank: hip_range is HIGH.
    # ============================================================
    if avg_torso_vert < 0.10:     scores["LEG RAISE"] += 25
    if torso_vert_std < 0.03:     scores["LEG RAISE"] += 30
    if torso_vert_std < 0.02:     scores["LEG RAISE"] += 15
    if hip_range > 40:            scores["LEG RAISE"] += 30
    if hip_range > 60:            scores["LEG RAISE"] += 15
    if elbow_range < 20:          scores["LEG RAISE"] += 20
    if avg_knee > 140:            scores["LEG RAISE"] += 10
    # Penalties
    if avg_torso_vert > 0.15:     scores["LEG RAISE"] -= 40
    if torso_vert_std > 0.05:     scores["LEG RAISE"] -= 30
    if elbow_range > 50:          scores["LEG RAISE"] -= 30
    if wrist_above_pct > 60:      scores["LEG RAISE"] -= 25

    # ============================================================
    #  DEAD HANG
    #  Hanging from bar. Arms FULLY EXTENDED. Wrists overhead.
    #  Elbows DO NOT BEND. Key separator from pull up: elbow static.
    # ============================================================
    if wrist_above_pct > 80:      scores["DEAD HANG"] += 30
    if wrist_above_pct > 90:      scores["DEAD HANG"] += 20
    if elbow_range < 15:          scores["DEAD HANG"] += 35
    if elbow_range < 8:           scores["DEAD HANG"] += 20
    if avg_elbow > 150:           scores["DEAD HANG"] += 25
    if avg_elbow > 160:           scores["DEAD HANG"] += 15
    if knee_range < 20:           scores["DEAD HANG"] += 15
    # Penalties
    if wrist_above_pct < 60:      scores["DEAD HANG"] -= 40
    if elbow_range > 30:          scores["DEAD HANG"] -= 40
    if elbow_range > 50:          scores["DEAD HANG"] -= 20
    if avg_elbow < 120:           scores["DEAD HANG"] -= 30
    if knee_range > 40:           scores["DEAD HANG"] -= 25

    # ============================================================
    #  BENT OVER ROW
    #  Torso hinged. STAYS hinged all set. Torso_vert_std LOW.
    #  Elbows pull back hard each rep = LARGE elbow range.
    #  Key separator from deadlift: elbow_range HIGH + torso static.
    # ============================================================
    if avg_torso_vert < 0.22:     scores["BENT OVER ROW"] += 25
    if avg_torso_lean > 25:       scores["BENT OVER ROW"] += 20
    if avg_torso_lean > 35:       scores["BENT OVER ROW"] += 10
    # Elbow pull = strongest signal
    if elbow_range > 35:          scores["BENT OVER ROW"] += 30
    if elbow_range > 55:          scores["BENT OVER ROW"] += 20
    if elbow_range > 70:          scores["BENT OVER ROW"] += 10
    # Knee stability
    if knee_range < 25:           scores["BENT OVER ROW"] += 20
    if knee_range < 15:           scores["BENT OVER ROW"] += 15
    if avg_knee > 140:            scores["BENT OVER ROW"] += 10
    # Torso stays hinged — never stands up
    if torso_vert_std < 0.04:     scores["BENT OVER ROW"] += 25
    if torso_vert_std < 0.02:     scores["BENT OVER ROW"] += 15
    if torso_vert_range < 0.10:   scores["BENT OVER ROW"] += 20
    if torso_vert_range < 0.06:   scores["BENT OVER ROW"] += 15
    # Penalties
    if elbow_range < 20:          scores["BENT OVER ROW"] -= 45
    if elbow_range < 12:          scores["BENT OVER ROW"] -= 20
    if avg_torso_vert > 0.28:     scores["BENT OVER ROW"] -= 30
    if knee_range > 50:           scores["BENT OVER ROW"] -= 35
    if torso_vert_range > 0.15:   scores["BENT OVER ROW"] -= 30
    if torso_vert_range > 0.20:   scores["BENT OVER ROW"] -= 20
    if knee_range > 80:           scores["BENT OVER ROW"] -= 40
    if wrist_above_pct > 60:      scores["BENT OVER ROW"] -= 25

    # ============================================================
    #  OVERHEAD PRESS
    #  Standing upright. Wrists go above head. Significant elbow range.
    # ============================================================
    if wrist_above_pct > 40:      scores["OVERHEAD PRESS"] += 30
    if wrist_above_pct > 60:      scores["OVERHEAD PRESS"] += 20
    if elbow_range > 40:          scores["OVERHEAD PRESS"] += 20
    if min_elbow < 90:            scores["OVERHEAD PRESS"] += 10
    if avg_knee > 150:            scores["OVERHEAD PRESS"] += 10
    if avg_torso_vert > 0.18:     scores["OVERHEAD PRESS"] += 10
    if avg_torso_vert > 0.22:     scores["OVERHEAD PRESS"] += 10
    # Penalties
    if wrist_above_pct < 20:      scores["OVERHEAD PRESS"] -= 35
    if knee_range > 50:           scores["OVERHEAD PRESS"] -= 25
    if knee_range > 80:           scores["OVERHEAD PRESS"] -= 40
    if elbow_range < 20:          scores["OVERHEAD PRESS"] -= 30
    if avg_torso_vert < 0.10:     scores["OVERHEAD PRESS"] -= 40

    # ---- DETECT WINNER / RESOLVE REPORT LABEL ----
    model_detected = max(scores, key=scores.get)
    model_confidence = scores[model_detected]
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    if requested_exercise:
        detected = requested_exercise.upper()
        confidence = scores.get(detected, 0)
        label_source = "USER SELECTION"
    else:
        detected = model_detected if model_confidence >= 25 else "UNKNOWN"
        confidence = model_confidence if detected != "UNKNOWN" else 0
        label_source = "AUTOMATIC PATTERN DETECTION"

    # ============================================================
    #  PRINT REPORT
    # ============================================================

    print("\n" + "=" * 62)
    print("          FITTECH — BIOMECHANICAL ANALYSIS REPORT")
    print("=" * 62)
    print(f"\n  ANALYZED EXERCISE  : {detected}")
    print(f"  LABEL SOURCE       : {label_source}")
    if requested_exercise:
        print(f"  MODEL SUGGESTION   : {model_detected} ({model_confidence} points)")
        print(f"  SELECTED SCORE     : {confidence} points (diagnostic only)")
    else:
        print(f"  CONFIDENCE SCORE   : {confidence} / 100")
    print(f"  FRAMES ANALYZED    : {total_frames}")

    print("\n" + "-" * 62)
    print("  KEY MEASUREMENTS")
    print("-" * 62)
    print(f"  Knee Angle         : Avg {avg_knee:.1f}  Min {min_knee:.1f}  Range {knee_range:.1f}")
    print(f"  Knee Asymmetry     : {knee_asymmetry:.1f}  (L: {avg_left_knee:.1f}  R: {avg_right_knee:.1f})")
    print(f"  Hip Angle          : Avg {avg_hip:.1f}  Min {min_hip:.1f}  Range {hip_range:.1f}")
    print(f"  Elbow Angle        : Avg {avg_elbow:.1f}  Min {min_elbow:.1f}  Range {elbow_range:.1f}")
    print(f"  Shoulder Abduction : Avg {avg_shoulder_abduct:.1f}  Max {max_shoulder_abduct:.1f}")
    print(f"  Torso Position     : Vertical {avg_torso_vert:.3f}  Lean {avg_torso_lean:.1f}")
    print(f"  Torso Stability    : Std {torso_vert_std:.3f}  Range {torso_vert_range:.3f}")
    print(f"  Hip to Knee Ratio  : {hip_to_knee_ratio:.2f}")
    print(f"  Wrist Above Shldr  : {wrist_above_pct:.1f} percent of frames")

    # ---- Score table ----
    print("\n" + "-" * 62)
    print("  ALL EXERCISE SCORES")
    print("-" * 62)
    for ex, sc in sorted_scores:
        bar    = "#" * max(0, sc // 4)
        marker = " <-- DETECTED" if ex == detected else ""
        print(f"  {ex:<22} : {sc:>3}  {bar}{marker}")

    # ---- Confusion feedback ----
    runners_up = [
        (ex, sc) for ex, sc in sorted_scores
        if ex != detected and sc >= confidence - 22 and sc > 15
    ]

    if runners_up and detected != "UNKNOWN":
        print("\n" + "-" * 62)
        print("  FORM SIMILARITY NOTICE")
        print("-" * 62)
        for ex, sc in runners_up[:2]:
            pair_key = frozenset({detected, ex})
            note     = CONFUSION_NOTES.get(pair_key)
            print(f"\n  Your form also resembles  : {ex}  (score: {sc})")
            if note:
                print(f"  Body part causing overlap : {note['body_part']}")
                print(f"  Why this happens          : {note['reason']}")
                print(f"  How to fix it             : {note['fix']}")
            else:
                print(f"  Some measurements overlap between {detected} and {ex}.")

    # ============================================================
    #  FEEDBACK PER EXERCISE
    # ============================================================

    print("\n" + "-" * 62)
    print("  FORM FEEDBACK AND VERDICT")
    print("-" * 62)

    if detected == "SQUAT":
        print("\n  SQUAT ANALYSIS\n")
        print(f"  Depth              : ", end="")
        if min_knee <= 90:    print("EXCELLENT — Below parallel")
        elif min_knee <= 100: print("GOOD — Parallel reached")
        elif min_knee <= 115: print("AVERAGE — Close but not quite")
        else:                 print("POOR — Not reaching parallel")

        print(f"  Torso              : ", end="")
        if avg_hip >= 70:   print("GOOD — Upright")
        elif avg_hip >= 55: print("MODERATE — Some forward lean")
        else:               print("POOR — Excessive forward lean")

        print(f"  Symmetry           : ", end="")
        if knee_asymmetry < 10:   print("GOOD — Both legs even")
        elif knee_asymmetry < 20: print("SLIGHT — Minor asymmetry")
        else:                     print("POOR — Significant imbalance")

        print()
        if min_knee <= 90 and avg_hip >= 70 and knee_asymmetry < 10:
            print("  VERDICT: Excellent squat form. Full depth, upright torso, symmetric drive. Focus on progressive overload to keep improving.")
        elif min_knee <= 90 and avg_hip < 55:
            print("  VERDICT: Great depth but too much forward lean. This shifts load from quads and glutes to the lower back — injury risk over time. Brace core hard before every descent, keep chest up, work on thoracic spine mobility daily.")
        elif min_knee > 115 and min_knee <= 130:
            print("  VERDICT: Not reaching parallel. Significantly reduces glute and quad activation. Stretch ankles daily, work on hip flexor mobility, try goblet squats to build the depth habit.")
        elif min_knee > 130:
            print("  VERDICT: Very shallow squat — barely activates legs. Remove all weight and practice bodyweight squats to full depth every day. Ankle and hip mobility work is essential before loading again.")
        elif knee_asymmetry > 20:
            print("  VERDICT: Significant left-right imbalance. One leg is dominant. Add Bulgarian split squats and single-leg press to balance both sides.")
        else:
            print("  VERDICT: Solid squat with room to improve. Depth first — weight is secondary.")

    elif detected == "DEADLIFT":
        print("\n  DEADLIFT ANALYSIS\n")
        print(f"  Hip Hinge          : ", end="")
        if min_hip < 40:    print("EXCELLENT — Strong hinge")
        elif min_hip < 55:  print("GOOD — Adequate hinge")
        else:               print("WEAK — Insufficient hinge")

        print(f"  Knee Control       : ", end="")
        if knee_range < 25:   print("GOOD — Stable knees")
        elif knee_range < 45: print("MODERATE — Some extra bend")
        else:                 print("POOR — Too much knee bend")

        print(f"  Hip Dominance      : ", end="")
        if hip_to_knee_ratio > 1.5: print(f"GOOD ({hip_to_knee_ratio:.2f})")
        else:                       print(f"WEAK ({hip_to_knee_ratio:.2f}) — knees too involved")

        print()
        if min_hip < 45 and knee_range < 45 and hip_to_knee_ratio > 1.2:
            print("  VERDICT: Good deadlift mechanics. Hip hinge is strong and knees are controlled. Keep back completely flat throughout the pull. Drive through heels, bar close to shins, lock out with glutes squeezed hard at the top.")
        elif min_hip >= 55:
            print("  VERDICT: Hip hinge is insufficient. You are squatting the weight up. Push hips back before pulling. Practice hip hinge wall drill — stand a foot from wall and push hips back until they touch it.")
        elif knee_range > 55:
            print("  VERDICT: Too much knee bend — deadlift becoming squat-like. Conventional deadlift uses only about 35 degrees of knee flexion. Set hips above knee level, keep shins more vertical, initiate with hip push not knee bend.")
        else:
            print("  VERDICT: Decent deadlift. Refine setup — hips above knees, shoulders in front of bar, chest up, lats engaged.")

    elif detected == "LUNGE":
        print("\n  LUNGE ANALYSIS\n")
        print(f"  Depth              : ", end="")
        if min_knee <= 90:    print("EXCELLENT — Full depth")
        elif min_knee <= 110: print("GOOD")
        else:                 print("POOR — Not deep enough")

        print()
        if min_knee <= 100:
            print("  VERDICT: Good lunge depth. Front knee tracks over ankle — not caving inward. Torso upright throughout. Back knee nearly touches the ground. Train both legs equally to avoid imbalances.")
        elif min_knee <= 120:
            print("  VERDICT: Getting close to proper depth. Take a longer stride so front thigh reaches parallel to floor. Lower hips straight down, keep torso vertical.")
        else:
            print("  VERDICT: Insufficient depth. Front thigh should be parallel to floor at the bottom — that is where maximum glute and quad activation happens. Focus on depth with bodyweight before adding load.")

    elif detected == "PUSH UP":
        print("\n  PUSH UP ANALYSIS\n")
        print(f"  Depth              : ", end="")
        if min_elbow < 80:    print("EXCELLENT — Chest near floor")
        elif min_elbow < 100: print("GOOD")
        else:                 print("POOR — Not deep enough")

        print(f"  Range of Motion    : ", end="")
        if elbow_range > 70:   print("EXCELLENT")
        elif elbow_range > 50: print("GOOD")
        else:                  print("POOR — Limited range")

        print()
        if min_elbow < 90 and elbow_range > 65:
            print("  VERDICT: Strong push up form. Good depth and range. Add a 3 second lowering phase on every rep — massively increases difficulty. Elbows at 45 degrees from body. Rigid straight line from head to heels throughout.")
        elif min_elbow >= 90:
            print("  VERDICT: Not deep enough. Chest should nearly touch the floor on every rep. Half reps build half the muscle. Drop to knee push ups if needed — full range with less load always beats partial range with more.")
        else:
            print("  VERDICT: Limited range of motion. Consciously think about touching chest to the floor on every rep. Stretch chest and shoulders daily.")

    elif detected == "PLANK":
        print("\n  PLANK ANALYSIS\n")
        print(f"  Body Stability     : ", end="")
        if torso_vert_std < 0.02:   print("EXCELLENT — Very stable hold")
        elif torso_vert_std < 0.04: print("GOOD — Mostly stable")
        else:                       print("WEAK — Too much movement")

        print()
        if torso_vert_std < 0.02:
            print("  VERDICT: Excellent plank stability. Body is rigid and controlled. Progress by increasing hold time by 10 seconds each session. Add shoulder taps or leg lifts for advanced variation. Never hold beyond the point where your hips start sagging.")
        else:
            print("  VERDICT: Too much movement detected. Squeeze glutes, brace core hard, tuck pelvis slightly, keep neck neutral. Your body should form one perfectly straight line from head to heels. If hips are sagging or rising the hold is too long — shorten duration and focus on quality over quantity.")

    elif detected == "PULL UP":
        print("\n  PULL UP ANALYSIS\n")
        print(f"  Elbow Depth        : ", end="")
        if min_elbow < 70:    print("EXCELLENT — Full pull, chin over bar")
        elif min_elbow < 90:  print("GOOD — Strong pull")
        else:                 print("POOR — Not pulling high enough")

        print(f"  Range of Motion    : ", end="")
        if elbow_range > 80:   print("EXCELLENT")
        elif elbow_range > 60: print("GOOD")
        else:                  print("POOR — Not using full range")

        print()
        if min_elbow < 80 and elbow_range > 70:
            print("  VERDICT: Strong pull up mechanics. Good range of motion. Drive your elbows down and back — think of pulling the bar to your chest rather than pulling yourself to the bar. Squeeze shoulder blades together at the top. Start from a full dead hang at the bottom of every rep for maximum lat activation.")
        elif min_elbow >= 90:
            print("  VERDICT: Not pulling high enough. Chin should clear the bar at the top of every rep. Work on band-assisted pull ups or lat pulldowns to build the specific strength. Dead hangs and scapular pulls are essential foundation work.")
        else:
            print("  VERDICT: Not using the full range. Start each rep from a completely dead hang — arms fully extended. Partial reps where you skip the bottom significantly reduce lat activation and limit progress.")

    elif detected == "SIT-UP":
        print("\n  SIT-UP ANALYSIS\n")
        print(f"  Range of Motion    : ", end="")
        if torso_vert_range > 0.22:   print("GOOD — Full range")
        elif torso_vert_range > 0.15: print("MODERATE — Partial range")
        else:                         print("POOR — Very limited range")

        print()
        if torso_vert_range > 0.18:
            print("  VERDICT: Good sit-up range of motion. To protect your lower back, slow the lowering phase and let your spine touch the floor segment by segment. Keep your feet anchored and core engaged throughout. If you feel it mostly in your hip flexors rather than your abs, try crunches instead for better core isolation.")
        else:
            print("  VERDICT: Not coming up fully or not going back down completely. Full range means torso reaches near vertical at the top and returns all the way to the floor at the bottom. Work on hip flexor flexibility if coming up fully feels restricted.")

    elif detected == "LEG RAISE":
        print("\n  LEG RAISE ANALYSIS\n")
        print(f"  Hip Range          : ", end="")
        if hip_range > 60:    print("EXCELLENT — Full leg raise")
        elif hip_range > 40:  print("GOOD")
        else:                 print("POOR — Not raising legs high enough")

        print(f"  Torso Stability    : ", end="")
        if torso_vert_std < 0.02:   print("EXCELLENT — Completely flat")
        elif torso_vert_std < 0.04: print("GOOD — Mostly stable")
        else:                       print("POOR — Torso lifting off floor")

        print()
        if hip_range > 50 and torso_vert_std < 0.03:
            print("  VERDICT: Good leg raise mechanics. Good height with a stable torso. Keep your lower back pressed completely flat into the floor throughout — the moment it arches and lifts off you have lost core engagement. Add a 2 second pause at the top for extra difficulty.")
        elif torso_vert_std > 0.05:
            print("  VERDICT: Torso is lifting off the floor. Your core is not strong enough yet to control the lever created by your legs. Shorten the range of motion — only raise legs as high as you can while keeping lower back fully pressed to the floor. Build up gradually.")
        else:
            print("  VERDICT: Not raising legs high enough. Aim to bring them perpendicular to the floor. Keep legs relatively straight and lower them slowly — the lowering phase is where most of the core work happens.")

    elif detected == "DEAD HANG":
        print("\n  DEAD HANG ANALYSIS\n")
        print(f"  Arm Extension      : ", end="")
        if avg_elbow > 160:   print("EXCELLENT — Arms fully extended")
        elif avg_elbow > 150: print("GOOD — Near full extension")
        else:                 print("POOR — Arms not fully straight")

        print(f"  Body Stability     : ", end="")
        if knee_range < 10:   print("EXCELLENT — Minimal swing")
        elif knee_range < 20: print("GOOD — Small swing")
        else:                 print("POOR — Too much swing")

        print()
        if avg_elbow > 155 and knee_range < 15:
            print("  VERDICT: Good dead hang. Arms fully extended and body controlled. Dead hang is one of the best exercises for shoulder health, grip strength, and spinal decompression. Work up to 60 second holds. While hanging, actively depress your shoulder blades — pull them down and back slightly — this protects the shoulder joint and builds the foundation needed for pull ups.")
        elif avg_elbow < 145:
            print("  VERDICT: Arms are not fully extending. In a dead hang your arms should be completely straight with elbows fully locked out. Hanging with bent arms defeats the purpose and puts unnecessary tension in the biceps. Relax completely and let gravity pull you to full arm extension.")
        else:
            print("  VERDICT: Good position. Minimize swinging by crossing your feet behind you and squeezing glutes and core to stabilize. Build hang duration progressively and focus on deep breathing while hanging.")

    elif detected == "BENT OVER ROW":
        print("\n  BENT OVER ROW ANALYSIS\n")
        print(f"  Torso Position     : ", end="")
        if avg_torso_vert < 0.15:   print("EXCELLENT — Strong forward lean")
        elif avg_torso_vert < 0.22: print("GOOD — Adequate angle")
        else:                       print("POOR — Torso too upright")

        print(f"  Pull Range         : ", end="")
        if elbow_range > 60:   print("GOOD — Strong pull")
        elif elbow_range > 40: print("MODERATE — Adequate")
        else:                  print("POOR — Limited pull range")

        print()
        if avg_torso_vert < 0.20 and elbow_range > 45:
            print("  VERDICT: Good bent over row mechanics. Pull toward your lower chest and squeeze shoulder blades together hard at the top — hold for one second. Back completely flat throughout — rounding under load is the most dangerous error in this movement. Core braced on every single rep.")
        elif avg_torso_vert > 0.25:
            print("  VERDICT: Torso too upright. Hinge at the hip until torso is roughly 45 degrees from vertical. An upright torso row barely activates lats and turns into a bicep exercise. Hinge forward, back flat, then pull.")
        else:
            print("  VERDICT: Work on elbow pull range. Elbow should drive well behind your torso at the top. Reduce weight slightly and add lat stretching to your routine.")

    elif detected == "OVERHEAD PRESS":
        print("\n  OVERHEAD PRESS ANALYSIS\n")
        print(f"  Overhead Reach     : ", end="")
        if wrist_above_pct > 60:   print("GOOD — Wrists reaching overhead")
        elif wrist_above_pct > 40: print("MODERATE — Partial overhead")
        else:                      print("POOR — Not reaching overhead")

        print(f"  Range of Motion    : ", end="")
        if elbow_range > 60:   print("EXCELLENT")
        elif elbow_range > 40: print("GOOD")
        else:                  print("POOR — Limited range")

        print()
        if wrist_above_pct > 55 and elbow_range > 50:
            print("  VERDICT: Good overhead press mechanics. Fully lock elbows at the top of every rep. At the bottom elbows should be at 90 degrees or below. Do not arch your lower back to get the weight overhead — if you are arching significantly the weight is too heavy.")
        elif wrist_above_pct < 35:
            print("  VERDICT: Arms not reaching overhead consistently. Reduce weight and press all the way to full arm extension on every rep. Limited overhead range usually means tight lats — add lat and overhead stretching to warm-up daily.")
        else:
            print("  VERDICT: Range of motion needs work. Full range is shoulder height to complete overhead lockout. Focus on the top lockout every rep.")

    else:
        print("\n  Could not confidently detect a specific exercise.")
        print(f"  Highest score: {confidence} — below minimum threshold of 25.")
        print()
        print("  Tips for better detection:")
        print("  - Record from SIDE ANGLE — most important factor")
        print("  - Full body visible in frame throughout the entire video")
        print("  - Good lighting — avoid backlit or dark environments")
        print("  - Controlled pace — very fast reps reduce landmark accuracy")
        print()
        print("  Supported exercises:")
        print("  squat | deadlift | lunge")
        print("  push up | plank | sit up | leg raise")
        print("  pull up | dead hang")
        print("  bent over row | overhead press")

    print("\n" + "=" * 62 + "\n")


# ============================================================
#  MAIN MENU LOOP
# ============================================================

print("\n" + "=" * 62)
print("          FITTECH — AI BIOMECHANICAL ANALYZER v5.0")
print("=" * 62)
print("\n  Supported exercises:")
print()
print("  LOWER BODY  : squat | deadlift | lunge")
print("  FLOOR       : push up | plank | sit up | leg raise")
print("  HANGING     : pull up | dead hang")
print("  UPPER BODY  : bent over row | overhead press")
print()
print("  File name tip: pushup/push_up/push-up all work")
print("                 bentoverrow/row/rows all work")
print("                 overheadpress/ohp/shoulderpress all work")

while True:
    print("\n" + "-" * 62)
    user_input = input("  Enter exercise name (or 'exit' to quit): ").strip().lower()

    if user_input == "exit":
        print("\n  Session complete. Train hard, train smart.\n")
        break

    if user_input in VIDEO_ALIASES:
        aliases    = VIDEO_ALIASES[user_input]
        video_path = find_video(aliases)

        if video_path:
            print(f"\n  Found: {video_path}")
            analyze_video(video_path, requested_exercise=user_input)
        else:
            print(f"\n  ERROR: No video found for '{user_input}'.")
            print(f"  Searched for: {', '.join(a + '.mp4' for a in aliases[:4])} etc.")
            print("  Make sure the video file is in the BIOMECH folder.")
            print("  Any of these names work: " + " | ".join(aliases[:5]))
    else:
        print("\n  Not recognized. Try one of the exercise names shown above.")