import cv2
import mediapipe as mp
import numpy as np
import os

# ============================================================
#  FORMIQ — BIOMECHANICAL ANALYZER v4.0 (Backend Engine)
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

# ---- Smooth Signal ----
def smooth_series(series, window=7):
    arr = np.array(series, dtype=float)
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode='same')

def find_video(base_names, search_dirs=None):
    if search_dirs is None:
        search_dirs = [".", "backend/sample_videos", "backend/uploads", "../backend/sample_videos"]
    extensions = [".mp4", ".MP4", ".mov", ".MOV", ".avi", ".AVI"]
    for d in search_dirs:
        for name in base_names:
            for ext in extensions:
                path = os.path.join(d, name + ext)
                if os.path.exists(path):
                    return path
    return None

def resolve_video_path(user_file_input, base_names=None):
    if user_file_input and os.path.exists(user_file_input):
        return user_file_input
    if user_file_input:
        extensions = [".mp4", ".MP4", ".mov", ".MOV", ".avi", ".AVI"]
        for ext in extensions:
            path = user_file_input + ext
            if os.path.exists(path):
                return path
    if base_names:
        return find_video(base_names)
    return None

VIDEO_ALIASES = {
    "squat"          : ["squat", "squats", "squat_video", "Squat"],
    "deadlift"       : ["deadlift", "deadlifts", "dead_lift", "Deadlift"],
    "rdl"            : ["rdl", "RDL", "romanian", "romanian_deadlift", "romaniandl"],
    "lunge"          : ["lunge", "lunges", "Lunge"],
    "calf raise"     : ["calfraise", "calf_raise", "calfrise", "calf raise", "Calfraise"],
    "push up"        : ["pushup", "push_up", "push-up", "pushups", "PushUp", "Pushup", "push up"],
    "shoulder press" : ["shoulderpress", "shoulder_press", "ShoulderPress", "shoulder press"],
    "bicep curl"     : ["bicepcurl", "bicep_curl", "bicep curl", "BicepCurl", "curl"],
    "lateral raise"  : ["lateralraise", "lateral_raise", "lateral raise", "LateralRaise"],
    "bent over row"  : ["bentoverrow", "bent_over_row", "bentover_row", "BentOverRow",
                        "bent over row", "row", "rows", "bent"],
}

CONFUSION_NOTES = {
    frozenset({"SQUAT", "DEADLIFT"}): {
        "body_part": "Knee angle and hip hinge ratio",
        "reason": "Your knee bent more than a typical deadlift allows.",
        "fix": "For deadlift: Set up with hips ABOVE knee level. Push knees out slightly.",
    },
    frozenset({"DEADLIFT", "BENT OVER ROW"}): {
        "body_part": "Torso vertical stability across the set",
        "reason": "A deadlift stands fully upright after every single rep, while a row stays hinged.",
        "fix": "For deadlift: Stand completely upright at the top. For row: Stay hinged throughout.",
    },
}

def analyze_video(VIDEO_PATH):
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        err_msg = f"ERROR: Could not open '{VIDEO_PATH}'"
        print(err_msg)
        return {"error": err_msg}

    data = {
        'left_knee': [], 'right_knee': [],
        'left_hip': [], 'right_hip': [],
        'left_elbow': [], 'right_elbow': [],
        'left_shoulder': [], 'right_shoulder': [],
        'left_ankle': [], 'right_ankle': [],
        'torso_lean': [], 'torso_vertical': [],
        'wrist_above_shoulder': [],
        'left_wrist_y': [], 'right_wrist_y': [],
        'left_shoulder_y': [], 'right_shoulder_y': [],
        'hip_y': [], 'shoulder_y': [],
        'left_knee_y': [], 'right_knee_y': [],
        'left_ankle_y': [], 'right_ankle_y': [],
        'left_shoulder_abduct': [], 'right_shoulder_abduct': [],
        'knee_series': [], 'elbow_series': [], 'hip_series': [],
        'spine_series': [], 'left_spine': [], 'right_spine': [],
    }

    print(f"\n  Analyzing video '{VIDEO_PATH}'... please wait.")

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
                l_ear       = pt(mp_pose.PoseLandmark.LEFT_EAR)
                r_ear       = pt(mp_pose.PoseLandmark.RIGHT_EAR)

                mid_shoulder = [(l_shoulder[0]+r_shoulder[0])/2, (l_shoulder[1]+r_shoulder[1])/2]
                mid_hip      = [(l_hip[0]+r_hip[0])/2, (l_hip[1]+r_hip[1])/2]
                mid_ear      = [(l_ear[0]+r_ear[0])/2, (l_ear[1]+r_ear[1])/2]

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

                lsp = calculate_angle(l_hip, l_shoulder, l_ear)
                rsp = calculate_angle(r_hip, r_shoulder, r_ear)
                data['left_spine'].append(lsp)
                data['right_spine'].append(rsp)
                data['spine_series'].append((lsp + rsp) / 2)

                data['torso_lean'].append(vertical_angle(mid_shoulder, mid_hip))
                data['torso_vertical'].append(mid_hip[1] - mid_shoulder[1])

                avg_wrist_y    = (l_wrist[1] + r_wrist[1]) / 2
                avg_shoulder_y = (l_shoulder[1] + r_shoulder[1]) / 2
                data['wrist_above_shoulder'].append(1 if avg_wrist_y < avg_shoulder_y else 0)

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
        err_msg = "ERROR: No body detected in video."
        print(f"  {err_msg}")
        return {"error": err_msg}

    total_frames = len(data['left_knee'])

    smooth_keys = [
        'left_knee','right_knee','left_hip','right_hip',
        'left_elbow','right_elbow','left_shoulder','right_shoulder',
        'left_ankle','right_ankle','torso_lean','torso_vertical',
        'left_shoulder_abduct','right_shoulder_abduct',
        'knee_series','hip_series','elbow_series',
        'left_spine','right_spine','spine_series'
    ]
    for key in smooth_keys:
        data[key] = smooth_series(data[key])

    avg_left_knee    = float(np.mean(data['left_knee']))
    avg_right_knee   = float(np.mean(data['right_knee']))
    avg_knee         = float((avg_left_knee + avg_right_knee) / 2)
    min_knee         = float(min(np.min(data['left_knee']), np.min(data['right_knee'])))
    max_knee         = float(max(np.max(data['left_knee']), np.max(data['right_knee'])))
    knee_range       = float(max_knee - min_knee)
    knee_std         = float(np.std(data['knee_series']))
    knee_asymmetry   = float(abs(avg_left_knee - avg_right_knee))

    avg_left_hip     = float(np.mean(data['left_hip']))
    avg_right_hip    = float(np.mean(data['right_hip']))
    avg_hip          = float((avg_left_hip + avg_right_hip) / 2)
    min_hip          = float(min(np.min(data['left_hip']), np.min(data['right_hip'])))
    max_hip          = float(max(np.max(data['left_hip']), np.max(data['right_hip'])))
    hip_range        = float(max_hip - min_hip)

    avg_left_elbow   = float(np.mean(data['left_elbow']))
    avg_right_elbow  = float(np.mean(data['right_elbow']))
    avg_elbow        = float((avg_left_elbow + avg_right_elbow) / 2)
    min_elbow        = float(min(np.min(data['left_elbow']), np.min(data['right_elbow'])))
    max_elbow        = float(max(np.max(data['left_elbow']), np.max(data['right_elbow'])))
    elbow_range      = float(max_elbow - min_elbow)

    avg_shoulder     = float((np.mean(data['left_shoulder']) + np.mean(data['right_shoulder'])) / 2)
    avg_shoulder_abduct = float((np.mean(data['left_shoulder_abduct']) + np.mean(data['right_shoulder_abduct'])) / 2)
    max_shoulder_abduct = float(max(np.max(data['left_shoulder_abduct']), np.max(data['right_shoulder_abduct'])))

    avg_ankle        = float((np.mean(data['left_ankle']) + np.mean(data['right_ankle'])) / 2)
    avg_spine        = float(np.mean(data['spine_series']))
    min_spine        = float(np.min(data['spine_series']))

    avg_torso_lean   = float(np.mean(data['torso_lean']))
    avg_torso_vert   = float(np.mean(data['torso_vertical']))
    torso_vert_range = float(np.max(data['torso_vertical']) - np.min(data['torso_vertical']))
    torso_vert_std   = float(np.std(data['torso_vertical']))

    wrist_above_pct  = float(np.mean(data['wrist_above_shoulder']) * 100)
    knee_y_asymmetry = float(abs(np.mean(data['left_knee_y']) - np.mean(data['right_knee_y'])))
    hip_to_knee_ratio = float(hip_range / (knee_range + 1e-6))

    scores = {
        "SQUAT"            : 0,
        "DEADLIFT"         : 0,
        "ROMANIAN DEADLIFT": 0,
        "LUNGE"            : 0,
        "CALF RAISE"       : 0,
        "PUSH UP"          : 0,
        "SHOULDER PRESS"   : 0,
        "BICEP CURL"       : 0,
        "LATERAL RAISE"    : 0,
        "BENT OVER ROW"    : 0,
    }

    # SQUAT
    if knee_range > 60:           scores["SQUAT"] += 15
    if knee_range > 90:           scores["SQUAT"] += 10
    if min_knee < 130:            scores["SQUAT"] += 10
    if min_knee < 100:            scores["SQUAT"] += 10
    if min_knee < 90:             scores["SQUAT"] += 10
    if knee_asymmetry < 20:       scores["SQUAT"] += 10
    if avg_torso_vert > 0.22:     scores["SQUAT"] += 15
    if avg_torso_vert > 0.26:     scores["SQUAT"] += 5
    if knee_std > 15:             scores["SQUAT"] += 5
    if hip_to_knee_ratio < 0.8:   scores["SQUAT"] += 10
    if elbow_range > 80:          scores["SQUAT"] -= 20
    if avg_torso_vert < 0.15:     scores["SQUAT"] -= 40
    if avg_torso_vert < 0.10:     scores["SQUAT"] -= 30
    if knee_range < 30:           scores["SQUAT"] -= 35
    if avg_torso_lean > 30:       scores["SQUAT"] -= 25
    if torso_vert_range > 0.20:   scores["SQUAT"] -= 15

    # DEADLIFT
    if min_hip < 35:              scores["DEADLIFT"] += 25
    if hip_range > 100:           scores["DEADLIFT"] += 20
    if knee_range > 80:           scores["DEADLIFT"] += 15
    if avg_torso_vert < 0.25:     scores["DEADLIFT"] += 15
    if avg_torso_lean > 20:       scores["DEADLIFT"] += 15
    if hip_to_knee_ratio > 1.2:   scores["DEADLIFT"] += 10
    if min_knee < 100 and min_knee > 60: scores["DEADLIFT"] -= 20
    if avg_torso_vert > 0.30:     scores["DEADLIFT"] -= 25
    if torso_vert_std < 0.02:     scores["DEADLIFT"] -= 20
    if hip_to_knee_ratio < 0.7:   scores["DEADLIFT"] -= 20
    if avg_spine < 162:           scores["DEADLIFT"] -= 25

    # ROMANIAN DEADLIFT
    if avg_knee > 155:            scores["ROMANIAN DEADLIFT"] += 20
    if avg_knee > 163:            scores["ROMANIAN DEADLIFT"] += 10
    if knee_range < 20:           scores["ROMANIAN DEADLIFT"] += 20
    if knee_range < 10:           scores["ROMANIAN DEADLIFT"] += 10
    if min_hip < 70:              scores["ROMANIAN DEADLIFT"] += 15
    if min_hip < 55:              scores["ROMANIAN DEADLIFT"] += 10
    if hip_range > 30:            scores["ROMANIAN DEADLIFT"] += 5
    if hip_to_knee_ratio > 2.5:   scores["ROMANIAN DEADLIFT"] += 10
    if knee_range > 40:           scores["ROMANIAN DEADLIFT"] -= 35
    if avg_knee < 145:            scores["ROMANIAN DEADLIFT"] -= 25
    if min_knee < 110:            scores["ROMANIAN DEADLIFT"] -= 30
    if avg_spine < 162:           scores["ROMANIAN DEADLIFT"] -= 25

    # LUNGE
    if knee_asymmetry > 20:       scores["LUNGE"] += 25
    if knee_asymmetry > 35:       scores["LUNGE"] += 20
    if knee_y_asymmetry > 0.05:   scores["LUNGE"] += 20
    if min_knee < 130:            scores["LUNGE"] += 15
    if knee_range > 40:           scores["LUNGE"] += 10
    if avg_torso_vert > 0.15:     scores["LUNGE"] += 10
    if knee_asymmetry < 10:       scores["LUNGE"] -= 35
    if knee_y_asymmetry < 0.02:   scores["LUNGE"] -= 25

    # CALF RAISE
    if knee_range < 15:           scores["CALF RAISE"] += 30
    if knee_range < 8:            scores["CALF RAISE"] += 15
    if avg_knee > 160:            scores["CALF RAISE"] += 25
    if avg_torso_vert > 0.20:     scores["CALF RAISE"] += 15
    if avg_hip > 160:             scores["CALF RAISE"] += 15
    if knee_range > 25:           scores["CALF RAISE"] -= 35
    if elbow_range > 50:          scores["CALF RAISE"] -= 25
    if avg_torso_vert < 0.15:     scores["CALF RAISE"] -= 25

    # PUSH UP
    if avg_torso_vert < 0.10:     scores["PUSH UP"] += 30
    if avg_torso_vert < 0.06:     scores["PUSH UP"] += 15
    if elbow_range > 50:          scores["PUSH UP"] += 20
    if min_elbow < 100:           scores["PUSH UP"] += 15
    if min_elbow < 80:            scores["PUSH UP"] += 10
    if avg_knee > 150:            scores["PUSH UP"] += 10
    if avg_torso_vert > 0.15:     scores["PUSH UP"] -= 40
    if knee_range > 30:           scores["PUSH UP"] -= 25

    # SHOULDER PRESS
    if wrist_above_pct > 40:      scores["SHOULDER PRESS"] += 30
    if wrist_above_pct > 60:      scores["SHOULDER PRESS"] += 20
    if elbow_range > 40:          scores["SHOULDER PRESS"] += 20
    if min_elbow < 90:            scores["SHOULDER PRESS"] += 10
    if avg_knee > 150:            scores["SHOULDER PRESS"] += 10
    if avg_torso_vert > 0.18:     scores["SHOULDER PRESS"] += 10
    if wrist_above_pct < 20:      scores["SHOULDER PRESS"] -= 35
    if knee_range > 50:           scores["SHOULDER PRESS"] -= 25
    if knee_range > 80:           scores["SHOULDER PRESS"] -= 40

    # BICEP CURL
    if elbow_range > 70:          scores["BICEP CURL"] += 25
    if elbow_range > 90:          scores["BICEP CURL"] += 10
    if min_elbow < 60:            scores["BICEP CURL"] += 20
    if min_elbow < 45:            scores["BICEP CURL"] += 10
    if avg_knee > 150:            scores["BICEP CURL"] += 10
    if avg_torso_vert > 0.18:     scores["BICEP CURL"] += 5
    if avg_shoulder < 60:         scores["BICEP CURL"] += 10
    if wrist_above_pct < 30:      scores["BICEP CURL"] += 10
    if wrist_above_pct > 60:      scores["BICEP CURL"] -= 30
    if elbow_range < 40:          scores["BICEP CURL"] -= 30
    if knee_range > 40:           scores["BICEP CURL"] -= 40
    if knee_range > 80:           scores["BICEP CURL"] -= 40

    # LATERAL RAISE
    if avg_shoulder_abduct > 60:  scores["LATERAL RAISE"] += 25
    if max_shoulder_abduct > 80:  scores["LATERAL RAISE"] += 20
    if elbow_range > 20:          scores["LATERAL RAISE"] += 15
    if elbow_range < 60:          scores["LATERAL RAISE"] += 10
    if avg_knee > 150:            scores["LATERAL RAISE"] += 10
    if avg_torso_vert > 0.18:     scores["LATERAL RAISE"] += 10
    if wrist_above_pct < 50:      scores["LATERAL RAISE"] += 10
    if elbow_range > 80:          scores["LATERAL RAISE"] -= 25
    if wrist_above_pct > 60:      scores["LATERAL RAISE"] -= 25
    if avg_torso_vert < 0.12:     scores["LATERAL RAISE"] -= 25
    if knee_range > 80:           scores["LATERAL RAISE"] -= 40

    # BENT OVER ROW
    if avg_torso_vert < 0.22:     scores["BENT OVER ROW"] += 20
    if avg_torso_lean > 25:       scores["BENT OVER ROW"] += 15
    if elbow_range > 40:          scores["BENT OVER ROW"] += 20
    if elbow_range > 60:          scores["BENT OVER ROW"] += 20
    if min_knee > 70:             scores["BENT OVER ROW"] += 15
    if min_hip > 45:              scores["BENT OVER ROW"] += 10
    if min_knee < 60:             scores["BENT OVER ROW"] -= 50
    if min_hip < 30:              scores["BENT OVER ROW"] -= 40
    if avg_torso_vert > 0.28:     scores["BENT OVER ROW"] -= 30
    if wrist_above_pct > 50:      scores["BENT OVER ROW"] -= 25
    if elbow_range < 40:          scores["BENT OVER ROW"] -= 35
    if avg_spine < 162:           scores["BENT OVER ROW"] -= 25

    detected   = max(scores, key=scores.get)
    confidence = int(scores[detected])

    if confidence < 25:
        detected   = "UNKNOWN"
        confidence = 0

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # Form feedback & verdict text generation
    evaluations = []
    verdict_text = ""
    is_poor_form = False

    spine_status = "Flat & Neutral" if avg_spine >= 162 else "POOR — Rounded back detected"

    if detected == "BENT OVER ROW":
        t_eval = "EXCELLENT — Good forward lean" if avg_torso_vert < 0.15 else ("GOOD — Adequate torso angle" if avg_torso_vert < 0.22 else "POOR — Torso too upright")
        p_eval = "GOOD — Strong elbow pull" if elbow_range > 60 else ("MODERATE — Adequate pull" if elbow_range > 40 else "POOR — Limited pulling range")
        s_eval = "EXCELLENT — Flat neutral spine" if avg_spine >= 166 else ("GOOD — Acceptable spine posture" if avg_spine >= 162 else "POOR — [POOR FORM] Rounded back / spinal hunching detected")
        
        evaluations = [
            {"name": "Torso Position", "value": f"{avg_torso_lean:.1f}°", "status": t_eval},
            {"name": "Pull Range", "value": f"{elbow_range:.1f}°", "status": p_eval},
            {"name": "Spine Alignment", "value": f"{avg_spine:.1f}°", "status": s_eval},
        ]
        
        if avg_spine < 162:
            is_poor_form = True
            verdict_text = "[POOR FORM DETECTED] — Your upper/lower back is rounding forward. Rounding your spine under load creates dangerous shear stress on lumbar discs. Keep your chest up, pull shoulder blades back, brace your core hard, and maintain a flat spine throughout."
        elif avg_torso_vert < 0.20 and elbow_range > 45:
            verdict_text = "Good bent over row mechanics. Pull toward your lower chest and squeeze shoulder blades together hard at the top — hold for one second. Back completely flat throughout. Core braced on every single rep."
        elif avg_torso_vert > 0.25:
            verdict_text = "Torso too upright. Hinge at the hip until torso is roughly 45 degrees from vertical while keeping spine flat."
        else:
            verdict_text = "Work on elbow pull range. Drive elbows well behind your torso at the top for full lat activation."

    elif detected == "DEADLIFT":
        h_eval = "EXCELLENT — Strong hip hinge" if min_hip < 40 else ("GOOD — Adequate hip hinge" if min_hip < 55 else "WEAK — Insufficient hip hinge")
        k_eval = "GOOD — Knees stable" if knee_range < 25 else ("MODERATE — Some knee movement" if knee_range < 45 else "POOR — Too much knee bend")
        s_eval = "EXCELLENT — Flat neutral spine" if avg_spine >= 166 else ("GOOD — Acceptable spine posture" if avg_spine >= 162 else "POOR — Rounded back detected")
        
        evaluations = [
            {"name": "Hip Hinge Quality", "value": f"Min {min_hip:.1f}°", "status": h_eval},
            {"name": "Knee Control", "value": f"ROM {knee_range:.1f}°", "status": k_eval},
            {"name": "Spine Alignment", "value": f"{avg_spine:.1f}°", "status": s_eval},
        ]

        if avg_spine < 162:
            is_poor_form = True
            verdict_text = "[POOR FORM DETECTED] — Rounded back detected. Rounding your spine under deadlift load places extreme shear force on your lower lumbar discs. Brace your core, pull shoulders back, keep chest up, and maintain a flat spine."
        elif min_hip < 45 and knee_range < 45 and hip_to_knee_ratio > 1.2:
            verdict_text = "Good deadlift mechanics. Hip hinge is strong and knee movement is controlled. Keep back completely flat throughout the pull. Drive through heels, lock out glutes hard at top."
        else:
            verdict_text = "Refine setup — hips above knees, chest up, shoulders slightly in front of bar. Initiate with hip push not knee bend."

    else:
        evaluations = [
            {"name": "Knee Angle", "value": f"Avg {avg_knee:.1f}°", "status": "Analyzed"},
            {"name": "Hip Angle", "value": f"Avg {avg_hip:.1f}°", "status": "Analyzed"},
            {"name": "Spine Alignment", "value": f"Avg {avg_spine:.1f}°", "status": spine_status},
        ]
        verdict_text = f"Exercise detected as {detected}. Maintain proper brace and controlled tempo."

    report_payload = {
        "detected_exercise" : detected,
        "confidence_score"  : confidence,
        "total_frames"       : total_frames,
        "is_poor_form"       : is_poor_form,
        "key_measurements"   : {
            "knee_angle"         : {"avg": round(avg_knee, 1), "min": round(min_knee, 1), "range": round(knee_range, 1)},
            "knee_asymmetry"     : round(knee_asymmetry, 1),
            "hip_angle"          : {"avg": round(avg_hip, 1), "min": round(min_hip, 1), "range": round(hip_range, 1)},
            "elbow_angle"        : {"avg": round(avg_elbow, 1), "min": round(min_elbow, 1), "range": round(elbow_range, 1)},
            "shoulder_abduction" : {"avg": round(avg_shoulder_abduct, 1), "max": round(max_shoulder_abduct, 1)},
            "torso_position"     : {"vertical": round(avg_torso_vert, 3), "lean": round(avg_torso_lean, 1)},
            "torso_stability"    : {"std": round(torso_vert_std, 3), "range": round(torso_vert_range, 3)},
            "spine_alignment"    : {"avg": round(avg_spine, 1), "status": spine_status},
            "hip_knee_rom_ratio" : round(hip_to_knee_ratio, 2),
            "wrist_above_pct"    : round(wrist_above_pct, 1),
        },
        "all_scores"          : sorted_scores,
        "form_feedback"      : {
            "evaluations" : evaluations,
            "verdict"     : verdict_text
        }
    }

    return report_payload

if __name__ == "__main__":
    import sys
    target_video = sys.argv[1] if len(sys.argv) > 1 else "sample_videos/bent_over_row.mp4"
    res = analyze_video(target_video)
    print("\n  ANALYSIS COMPLETED FOR:", target_video)
    print("  DETECTED EXERCISE  :", res.get("detected_exercise"))
    print("  CONFIDENCE SCORE   :", res.get("confidence_score"))
    print("  VERDICT            :", res.get("form_feedback", {}).get("verdict"))
