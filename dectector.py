import cv2
import mediapipe as mp
import numpy as np
import os

# ============================================================
#  FORMIQ — BIOMECHANICAL ANALYZER v4.0
#  Research-grounded detection with confusion feedback
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

# ============================================================
#  SMART FILE FINDER
#  Tries multiple possible names so the user does not have to
#  rename files exactly. Checks the BIOMECH folder.
# ============================================================

def find_video(base_names):
    extensions = [".mp4", ".MP4", ".mov", ".MOV", ".avi", ".AVI"]
    for name in base_names:
        for ext in extensions:
            path = name + ext
            if os.path.exists(path):
                return path
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
                        "bent over row", "row", "rows"],
}

# ============================================================
#  CONFUSION NOTES
#  When two exercises score within 20 points of each other,
#  print which body part caused the confusion and how to fix it.
# ============================================================

CONFUSION_NOTES = {
    frozenset({"SQUAT", "DEADLIFT"}): {
        "body_part": "Knee angle and hip hinge ratio",
        "reason": (
            "Your knee bent more than a typical deadlift allows. "
            "Research shows a conventional deadlift setup has only about 35 degrees of "
            "knee flexion — meaning knee angle stays around 145 degrees or higher throughout. "
            "A squat goes all the way down to 90 degrees or below. "
            "If this was a deadlift, the extra knee bend means your hips are dropping too low "
            "and your setup is becoming squat-like."
        ),
        "fix": (
            "For deadlift: Set up with hips ABOVE knee level. Push knees out slightly, "
            "keep shins near vertical, and think about pushing the floor away. "
            "Your hip should be the primary hinge — not your knees."
        ),
    },
    frozenset({"DEADLIFT", "BENT OVER ROW"}): {
        "body_part": "Torso vertical stability across the set",
        "reason": (
            "Both exercises involve a hinged forward torso. "
            "The difference is what happens between reps — a deadlift stands fully upright "
            "after every single rep, so the torso cycles significantly from horizontal to vertical. "
            "A bent over row stays permanently hinged over for the entire set."
        ),
        "fix": (
            "For deadlift: Stand completely upright at the top of every rep — full hip extension, "
            "glutes squeezed. Do not do touch-and-go where you bounce off the floor — "
            "each rep should start from a dead stop with full posture reset. "
            "For bent over row: Stay hinged throughout the set. Never stand up between reps."
        ),
    },
    frozenset({"DEADLIFT", "ROMANIAN DEADLIFT"}): {
        "body_part": "Knee bend depth",
        "reason": (
            "Romanian deadlift research shows only 15 degrees of knee flexion — "
            "knee angle stays near 165 degrees throughout the entire movement. "
            "Your knee is bending more than this, which is pulling the detection toward "
            "conventional deadlift territory."
        ),
        "fix": (
            "For RDL: Keep knees almost completely straight throughout. "
            "Soft bend only — not a significant bend. The hip hinge does all the work. "
            "If your knees are bending significantly, you are turning it into a regular deadlift."
        ),
    },
    frozenset({"SQUAT", "LUNGE"}): {
        "body_part": "Knee symmetry between left and right leg",
        "reason": (
            "One of your knees is bending noticeably more than the other. "
            "In a proper squat both legs should work almost identically. "
            "The asymmetry is making the movement pattern look like a lunge."
        ),
        "fix": (
            "For squat: Check your stance width and foot position. "
            "If one side is consistently weaker or stiffer, add single-leg work "
            "like Bulgarian split squats to balance the two sides before squatting heavy."
        ),
    },
    frozenset({"BENT OVER ROW", "ROMANIAN DEADLIFT"}): {
        "body_part": "Elbow movement and torso hinge angle",
        "reason": (
            "Both exercises have a hinged torso and minimal knee bend. "
            "The difference is elbow movement — a row has significant elbow pull "
            "while an RDL has almost no arm movement at all."
        ),
        "fix": (
            "For bent over row: Make sure your elbows are actively pulling back and "
            "your shoulder blades are squeezing together at the top of every rep. "
            "For RDL: Your arms should hang dead straight throughout — "
            "no elbow bend at any point."
        ),
    },
    frozenset({"BICEP CURL", "SHOULDER PRESS"}): {
        "body_part": "Wrist final position and elbow travel path",
        "reason": (
            "Both involve significant elbow bending. "
            "The difference is where the wrists end up — "
            "a press finishes with wrists above the head, "
            "a curl stays at or below shoulder height."
        ),
        "fix": (
            "For shoulder press: Make sure you fully extend overhead on every rep. "
            "Wrists should clear the top of your head. "
            "For bicep curl: Keep elbows pinned to your sides — "
            "if wrists are going high your elbows are drifting forward."
        ),
    },
    frozenset({"LATERAL RAISE", "SHOULDER PRESS"}): {
        "body_part": "Wrist height and elbow range of motion",
        "reason": (
            "Both raise the arm above shoulder height but in different directions. "
            "A lateral raise moves out to the side, elbow stays bent, "
            "wrists reach shoulder height only. "
            "A press drives straight overhead with much larger elbow extension."
        ),
        "fix": (
            "For lateral raise: Stop at parallel — do not go above shoulder height. "
            "For shoulder press: Press all the way until arms are fully locked out overhead."
        ),
    },
}

# ============================================================
#  MAIN ANALYSIS FUNCTION
# ============================================================

def analyze_video(VIDEO_PATH):

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
        print("  ERROR: No body detected in video.")
        print("  Tips: Record from side angle. Full body visible. Good lighting.\n")
        return

    total_frames = len(data['left_knee'])

    # ---- Smooth all series ----
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

    avg_shoulder     = (np.mean(data['left_shoulder']) + np.mean(data['right_shoulder'])) / 2
    avg_shoulder_abduct = (np.mean(data['left_shoulder_abduct']) +
                           np.mean(data['right_shoulder_abduct'])) / 2
    max_shoulder_abduct = max(np.max(data['left_shoulder_abduct']),
                              np.max(data['right_shoulder_abduct']))

    avg_ankle        = (np.mean(data['left_ankle']) + np.mean(data['right_ankle'])) / 2

    avg_torso_lean   = np.mean(data['torso_lean'])
    avg_torso_vert   = np.mean(data['torso_vertical'])
    torso_vert_range = np.max(data['torso_vertical']) - np.min(data['torso_vertical'])
    torso_vert_std   = np.std(data['torso_vertical'])

    wrist_above_pct  = np.mean(data['wrist_above_shoulder']) * 100
    avg_wrist_y      = (np.mean(data['left_wrist_y']) + np.mean(data['right_wrist_y'])) / 2
    avg_shoulder_y   = (np.mean(data['left_shoulder_y']) + np.mean(data['right_shoulder_y'])) / 2

    knee_y_asymmetry = abs(np.mean(data['left_knee_y']) - np.mean(data['right_knee_y']))

    # ---- Key ratios (research-backed) ----
    # Deadlift is hip-dominant: hip_range >> knee_range
    # Squat is knee-dominant: knee_range >> hip_range
    hip_to_knee_ratio = hip_range / (knee_range + 1e-6)

    # Deadlift stands upright between reps: torso_vert_range is HIGH
    # Bent over row stays hinged all set: torso_vert_range is LOW
    # This is the single most reliable deadlift vs row separator

    # ============================================================
    #  CONFIDENCE SCORING — Every exercise scored 0-100
    # ============================================================

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

       # ============================================================
    #  SQUAT — research: knee dominant, torso upright throughout
    #  avg_torso_vert HIGH (>0.20), knee_range large, symmetric
    # ============================================================
    if knee_range > 60:           scores["SQUAT"] += 25
    if knee_range > 90:           scores["SQUAT"] += 10
    if min_knee < 130:            scores["SQUAT"] += 20
    if min_knee < 100:            scores["SQUAT"] += 15
    if min_knee < 90:             scores["SQUAT"] += 10
    if knee_asymmetry < 20:       scores["SQUAT"] += 15
    if avg_torso_vert > 0.22:     scores["SQUAT"] += 20  # upright torso = squat
    if avg_torso_vert > 0.26:     scores["SQUAT"] += 10  # very upright = strongly squat
    if knee_std > 15:             scores["SQUAT"] += 10
    if hip_to_knee_ratio < 0.8:   scores["SQUAT"] += 15  # knee dominant
    # PENALTIES
    if elbow_range > 80:          scores["SQUAT"] -= 20
    if avg_torso_vert < 0.15:     scores["SQUAT"] -= 40  # forward lean = NOT a squat
    if avg_torso_vert < 0.10:     scores["SQUAT"] -= 30  # horizontal = push up
    if knee_range < 30:           scores["SQUAT"] -= 35
    if avg_torso_lean > 30:       scores["SQUAT"] -= 25  # torso tilted = not squat
    if torso_vert_range > 0.20:   scores["SQUAT"] -= 15  # too much torso swing

    # ============================================================
    #  DEADLIFT — research: hip dominant, torso leans forward
    #  Knee flexion only ~35 degrees at setup = knee angle ~145
    #  Torso cycles from upright to forward and back each rep
    #  torso_vert_range HIGH (stands upright each rep)
    #  avg_torso_vert LOWER than squat (leans forward significantly)
    #  min_hip VERY LOW (hip hinge goes deep)
    # ============================================================
    if knee_range < 60:           scores["DEADLIFT"] += 20
    if knee_range < 40:           scores["DEADLIFT"] += 15
    if min_knee > 110:            scores["DEADLIFT"] += 15  # knee never squat-deep
    if min_knee > 125:            scores["DEADLIFT"] += 10
    if min_hip < 60:              scores["DEADLIFT"] += 25  # deep hip hinge
    if min_hip < 40:              scores["DEADLIFT"] += 20  # very deep hinge
    if min_hip < 30:              scores["DEADLIFT"] += 10
    if avg_torso_vert < 0.25:     scores["DEADLIFT"] += 15  # torso leans forward
    if avg_torso_vert < 0.18:     scores["DEADLIFT"] += 15  # strongly forward = deadlift
    if avg_torso_lean > 20:       scores["DEADLIFT"] += 20  # forward lean detected
    if avg_torso_lean > 35:       scores["DEADLIFT"] += 15  # strongly leaning = deadlift
    if hip_range > 40:            scores["DEADLIFT"] += 15  # large hip movement
    if hip_range > 60:            scores["DEADLIFT"] += 10
    if hip_to_knee_ratio > 1.2:   scores["DEADLIFT"] += 20  # hip dominant
    if hip_to_knee_ratio > 1.8:   scores["DEADLIFT"] += 10  # strongly hip dominant
    if torso_vert_range > 0.12:   scores["DEADLIFT"] += 20  # stands upright between reps
    if torso_vert_range > 0.18:   scores["DEADLIFT"] += 15  # strongly stands up each rep
    if knee_asymmetry < 15:       scores["DEADLIFT"] += 5
    # PENALTIES
    if min_knee < 100:            scores["DEADLIFT"] -= 35  # squat depth = not deadlift
    if min_knee < 90:             scores["DEADLIFT"] -= 20  # extra penalty for very deep
    if knee_range > 80:           scores["DEADLIFT"] -= 30  # too much knee = squat
    if avg_torso_vert > 0.30:     scores["DEADLIFT"] -= 25  # upright all time = not deadlift
    if torso_vert_std < 0.02:     scores["DEADLIFT"] -= 20  # static = row not deadlift
    if hip_to_knee_ratio < 0.7:   scores["DEADLIFT"] -= 20  # knee dominant = squat

    # ============================================================
    #  ROMANIAN DEADLIFT
    #  Research: ~15 deg knee flexion = knee angle ~165 deg throughout
    #  Knees stay nearly straight the entire set — very small knee range
    #  Pure hip hinge. Hamstring stretch at bottom.
    # ============================================================
    if avg_knee > 155:            scores["ROMANIAN DEADLIFT"] += 30  # knees very straight
    if avg_knee > 163:            scores["ROMANIAN DEADLIFT"] += 15
    if knee_range < 20:           scores["ROMANIAN DEADLIFT"] += 30
    if knee_range < 10:           scores["ROMANIAN DEADLIFT"] += 15
    if min_hip < 70:              scores["ROMANIAN DEADLIFT"] += 20
    if min_hip < 55:              scores["ROMANIAN DEADLIFT"] += 10
    if hip_range > 30:            scores["ROMANIAN DEADLIFT"] += 10
    if hip_to_knee_ratio > 2.5:   scores["ROMANIAN DEADLIFT"] += 15  # extremely hip-dominant
    # Penalties
    if knee_range > 40:           scores["ROMANIAN DEADLIFT"] -= 35  # too much knee bend
    if avg_knee < 145:            scores["ROMANIAN DEADLIFT"] -= 25  # squatting = not RDL
    if min_knee < 110:            scores["ROMANIAN DEADLIFT"] -= 30  # definitely not RDL

    # ============================================================
    #  LUNGE
    #  One knee bends deeply, other stays back = large L vs R asymmetry
    #  Left and right knee Y positions differ significantly
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

    # ============================================================
    #  CALF RAISE
    #  Knees completely still, standing upright, only ankle moves
    # ============================================================
    if knee_range < 15:           scores["CALF RAISE"] += 35
    if knee_range < 8:            scores["CALF RAISE"] += 20
    if avg_knee > 160:            scores["CALF RAISE"] += 25
    if avg_torso_vert > 0.20:     scores["CALF RAISE"] += 15
    if avg_hip > 160:             scores["CALF RAISE"] += 15
    # Penalties
    if knee_range > 25:           scores["CALF RAISE"] -= 35
    if elbow_range > 50:          scores["CALF RAISE"] -= 25
    if avg_torso_vert < 0.15:     scores["CALF RAISE"] -= 25

    # ============================================================
    #  PUSH UP
    #  Torso nearly horizontal throughout entire set
    #  Elbow bends from ~160 to below 90 — large elbow range
    #  Body in plank — torso_vertical very small (close to 0)
    # ============================================================
    if avg_torso_vert < 0.10:     scores["PUSH UP"] += 35
    if avg_torso_vert < 0.06:     scores["PUSH UP"] += 20
    if elbow_range > 50:          scores["PUSH UP"] += 25
    if min_elbow < 100:           scores["PUSH UP"] += 15
    if min_elbow < 80:            scores["PUSH UP"] += 15
    if avg_knee > 150:            scores["PUSH UP"] += 10
    # Penalties
    if avg_torso_vert > 0.15:     scores["PUSH UP"] -= 40  # upright = not a push up
    if knee_range > 30:           scores["PUSH UP"] -= 25

    # ============================================================
    #  SHOULDER PRESS
    #  Standing or seated. Wrists go above head most of the set.
    #  Significant elbow range, knees nearly still.
    # ============================================================
    if wrist_above_pct > 40:      scores["SHOULDER PRESS"] += 30
    if wrist_above_pct > 60:      scores["SHOULDER PRESS"] += 20
    if elbow_range > 40:          scores["SHOULDER PRESS"] += 20
    if min_elbow < 90:            scores["SHOULDER PRESS"] += 10
    if avg_knee > 150:            scores["SHOULDER PRESS"] += 10
    if avg_torso_vert > 0.18:     scores["SHOULDER PRESS"] += 10
    # Penalties
    if wrist_above_pct < 20:      scores["SHOULDER PRESS"] -= 35
    if knee_range > 50:           scores["SHOULDER PRESS"] -= 25
    if knee_range > 80:           scores["SHOULDER PRESS"] -= 40  # leg veto

    # ============================================================
    #  BICEP CURL
    #  Elbow goes from ~160 to ~30-45 degrees — massive elbow range
    #  Shoulder stays low. Wrists stay at or below shoulder level.
    #  Standing upright. Legs barely move.
    # ============================================================
    if elbow_range > 70:          scores["BICEP CURL"] += 30
    if elbow_range > 90:          scores["BICEP CURL"] += 15
    if min_elbow < 60:            scores["BICEP CURL"] += 25
    if min_elbow < 45:            scores["BICEP CURL"] += 10
    if avg_knee > 150:            scores["BICEP CURL"] += 10
    if avg_torso_vert > 0.18:     scores["BICEP CURL"] += 10
    if avg_shoulder < 60:         scores["BICEP CURL"] += 15
    if wrist_above_pct < 30:      scores["BICEP CURL"] += 15
    # Penalties
    if wrist_above_pct > 60:      scores["BICEP CURL"] -= 30  # wrist goes high = press
    if elbow_range < 40:          scores["BICEP CURL"] -= 30
    if knee_range > 40:           scores["BICEP CURL"] -= 40  # leg veto
    if knee_range > 80:           scores["BICEP CURL"] -= 40

    # ============================================================
    #  LATERAL RAISE
    #  Arm raises sideways to shoulder height. Elbow slightly bent.
    #  Shoulder abduction increases significantly.
    #  Wrists reach shoulder height but NOT above head.
    # ============================================================
    if avg_shoulder_abduct > 60:  scores["LATERAL RAISE"] += 25
    if max_shoulder_abduct > 80:  scores["LATERAL RAISE"] += 20
    if elbow_range > 20:          scores["LATERAL RAISE"] += 15
    if elbow_range < 60:          scores["LATERAL RAISE"] += 10
    if avg_knee > 150:            scores["LATERAL RAISE"] += 10
    if avg_torso_vert > 0.18:     scores["LATERAL RAISE"] += 10
    if wrist_above_pct < 50:      scores["LATERAL RAISE"] += 10
    # Penalties
    if elbow_range > 80:          scores["LATERAL RAISE"] -= 25  # too much elbow = curl
    if wrist_above_pct > 60:      scores["LATERAL RAISE"] -= 25  # overhead = press
    if avg_torso_vert < 0.12:     scores["LATERAL RAISE"] -= 25
    if knee_range > 80:           scores["LATERAL RAISE"] -= 40  # leg veto

    # ============================================================
    #  BENT OVER ROW
    #  Torso hinged 45-90 degrees forward, STAYS that way all set
    #  Elbow pulls back significantly each rep
    #  Knees barely move. Torso_vert_std LOW (static hinge).
    #  Torso_vert_range LOW (never stands upright between reps).
    #  This separates it from deadlift which stands up between reps.
    # ============================================================
    if avg_torso_vert < 0.22:     scores["BENT OVER ROW"] += 25
    if avg_torso_lean > 25:       scores["BENT OVER ROW"] += 20
    if elbow_range > 40:          scores["BENT OVER ROW"] += 20
    if knee_range < 25:           scores["BENT OVER ROW"] += 20
    if knee_range < 15:           scores["BENT OVER ROW"] += 10
    if avg_knee > 140:            scores["BENT OVER ROW"] += 10
    if torso_vert_std < 0.04:     scores["BENT OVER ROW"] += 25  # stays hinged = row
    if torso_vert_range < 0.10:   scores["BENT OVER ROW"] += 20  # never stands up = row
    # Penalties
    if avg_torso_vert > 0.28:     scores["BENT OVER ROW"] -= 30  # upright = not a row
    if knee_range > 50:           scores["BENT OVER ROW"] -= 35  # leg movement = deadlift
    if wrist_above_pct > 50:      scores["BENT OVER ROW"] -= 25
    if torso_vert_range > 0.15:   scores["BENT OVER ROW"] -= 30  # stands up = deadlift not row
    if knee_range > 80:           scores["BENT OVER ROW"] -= 40  # leg veto

    # ---- DETECT WINNER ----
    detected   = max(scores, key=scores.get)
    confidence = scores[detected]

    if confidence < 25:
        detected   = "UNKNOWN"
        confidence = 0

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # ============================================================
    #  PRINT REPORT
    # ============================================================

    print("\n" + "=" * 62)
    print("          FORMIQ — BIOMECHANICAL ANALYSIS REPORT")
    print("=" * 62)
    print(f"\n  DETECTED EXERCISE  : {detected}")
    print(f"  CONFIDENCE SCORE   : {confidence} / 100")
    print(f"  FRAMES ANALYZED    : {total_frames}")

    print("\n" + "-" * 62)
    print("  KEY MEASUREMENTS")
    print("-" * 62)
    print(f"  Knee Angle         : Avg {avg_knee:.1f}°  Min {min_knee:.1f}°  Range {knee_range:.1f}°")
    print(f"  Knee Asymmetry     : {knee_asymmetry:.1f}°  (L: {avg_left_knee:.1f}°  R: {avg_right_knee:.1f}°)")
    print(f"  Hip Angle          : Avg {avg_hip:.1f}°  Min {min_hip:.1f}°  Range {hip_range:.1f}°")
    print(f"  Elbow Angle        : Avg {avg_elbow:.1f}°  Min {min_elbow:.1f}°  Range {elbow_range:.1f}°")
    print(f"  Shoulder Abduction : Avg {avg_shoulder_abduct:.1f}°  Max {max_shoulder_abduct:.1f}°")
    print(f"  Torso Position     : Vertical {avg_torso_vert:.3f}  Lean {avg_torso_lean:.1f}°")
    print(f"  Torso Stability    : Std {torso_vert_std:.3f}  Range {torso_vert_range:.3f}")
    print(f"  Hip:Knee ROM ratio : {hip_to_knee_ratio:.2f}  (>1.2 = hip-dominant  <0.8 = knee-dominant)")
    print(f"  Wrist Above Shldr  : {wrist_above_pct:.1f}% of frames")

    # ---- Score table ----
    print("\n" + "-" * 62)
    print("  ALL EXERCISE SCORES")
    print("-" * 62)
    for ex, sc in sorted_scores:
        bar = "#" * max(0, sc // 4)
        marker = " <-- DETECTED" if ex == detected else ""
        print(f"  {ex:<22} : {sc:>3}  {bar}{marker}")

    # ---- Confusion feedback ----
    runners_up = [
        (ex, sc) for ex, sc in sorted_scores[1:]
        if ex != detected and sc >= confidence - 22 and sc > 15
    ]

    if runners_up and detected != "UNKNOWN":
        print("\n" + "-" * 62)
        print("  FORM SIMILARITY NOTICE")
        print("-" * 62)
        for ex, sc in runners_up[:2]:
            pair_key = frozenset({detected, ex})
            note     = CONFUSION_NOTES.get(pair_key)
            print(f"\n  Your form also resembles: {ex} (score: {sc})")
            if note:
                print(f"\n  Body part causing overlap : {note['body_part']}")
                print(f"\n  Why this happens          : {note['reason']}")
                print(f"\n  How to fix it             : {note['fix']}")
            else:
                print(f"  Some angle measurements overlap between {detected} and {ex}.")
                print("  This is normal for closely related exercises.")

    # ============================================================
    #  FEEDBACK PER EXERCISE
    # ============================================================

    print("\n" + "-" * 62)
    print("  FORM FEEDBACK AND VERDICT")
    print("-" * 62)

    if detected == "SQUAT":
        print("\n  SQUAT ANALYSIS\n")
        print(f"  Depth Assessment   : ", end="")
        if min_knee <= 90:
            print("EXCELLENT — Below parallel achieved")
        elif min_knee <= 100:
            print("GOOD — Parallel reached")
        elif min_knee <= 115:
            print("AVERAGE — Close to parallel but not quite")
        else:
            print("POOR — Not reaching parallel")

        print(f"  Torso Assessment   : ", end="")
        if avg_hip >= 70:
            print("GOOD — Upright torso maintained")
        elif avg_hip >= 55:
            print("MODERATE — Some forward lean")
        else:
            print("POOR — Excessive forward lean")

        print(f"  Symmetry           : ", end="")
        if knee_asymmetry < 10:
            print("GOOD — Both legs even")
        elif knee_asymmetry < 20:
            print("SLIGHT — Minor asymmetry")
        else:
            print("POOR — Significant imbalance")

        print()
        if min_knee <= 90 and avg_hip >= 70 and knee_asymmetry < 10:
            print("  VERDICT: Excellent squat form. Full depth with upright torso and symmetric drive. Focus on progressive overload to keep getting stronger.")
        elif min_knee <= 90 and avg_hip < 55:
            print("  VERDICT: Great depth but too much forward lean. This shifts load from quads and glutes to the lower back — injury risk over time. Brace core hard, keep chest up, work on thoracic spine mobility daily.")
        elif min_knee > 115 and min_knee <= 130:
            print("  VERDICT: Not reaching parallel. Reduces glute and quad activation significantly. Stretch ankles daily, work on hip flexor mobility, try goblet squats to build depth habit.")
        elif min_knee > 130:
            print("  VERDICT: Very shallow squat. Barely activates your legs. Remove all weight, practice bodyweight squats to full depth every day. Ankle and hip mobility work is essential before loading again.")
        elif knee_asymmetry > 20:
            print("  VERDICT: Significant left-right imbalance. One leg is dominant. Add Bulgarian split squats and single-leg press to balance the two sides.")
        else:
            print("  VERDICT: Solid squat with room to improve. Focus on depth first — weight is secondary.")

    elif detected == "DEADLIFT":
        print("\n  DEADLIFT ANALYSIS\n")
        print(f"  Hip Hinge Quality  : ", end="")
        if min_hip < 40:
            print("EXCELLENT — Strong hip hinge")
        elif min_hip < 55:
            print("GOOD — Adequate hip hinge")
        else:
            print("WEAK — Insufficient hip hinge")

        print(f"  Knee Control       : ", end="")
        if knee_range < 25:
            print("GOOD — Knees appropriately stable")
        elif knee_range < 45:
            print("MODERATE — Some extra knee movement")
        else:
            print("POOR — Too much knee bend, becoming squat-like")

        print(f"  Hip Dominance      : ", end="")
        if hip_to_knee_ratio > 1.5:
            print(f"GOOD — Hip-dominant pattern ({hip_to_knee_ratio:.2f})")
        else:
            print(f"WEAK — Not hip-dominant enough ({hip_to_knee_ratio:.2f})")

        print()
        if min_hip < 45 and knee_range < 45 and hip_to_knee_ratio > 1.2:
            print("  VERDICT: Good deadlift mechanics. Hip hinge is strong and knee movement is controlled. Keep back completely flat throughout the pull. Drive through heels, keep bar close to shins, lock out with glutes squeezed hard at top.")
        elif min_hip >= 55:
            print("  VERDICT: Hip hinge is insufficient. You are squatting the weight up rather than hinging. Stand close to bar, push hips back before pulling, think of pushing the floor away rather than pulling the bar up. Practice hip hinge wall drill until the pattern feels natural.")
        elif knee_range > 55:
            print("  VERDICT: Too much knee bend — deadlift is becoming squat-like. Research shows conventional deadlift should have only about 35 degrees of knee flexion at setup. Set hips above knee level, keep shins more vertical, initiate with hip push not knee bend.")
        else:
            print("  VERDICT: Decent deadlift. Refine your setup — hips above knees, shoulders slightly in front of bar, chest up, lats engaged. A perfect setup makes the pull feel almost automatic.")

    elif detected == "ROMANIAN DEADLIFT":
        print("\n  ROMANIAN DEADLIFT ANALYSIS\n")
        print(f"  Hip Hinge Quality  : ", end="")
        if min_hip < 50:
            print("EXCELLENT — Deep hamstring stretch")
        elif min_hip < 65:
            print("GOOD — Adequate range")
        else:
            print("WEAK — Not hinging enough")

        print(f"  Knee Control       : ", end="")
        if avg_knee > 160 and knee_range < 15:
            print("EXCELLENT — Knees nearly straight throughout")
        elif avg_knee > 155:
            print("GOOD — Knees appropriately extended")
        else:
            print("POOR — Too much knee bend for RDL")

        print()
        if min_hip < 55 and avg_knee > 155:
            print("  VERDICT: Solid RDL mechanics. Hip hinge strong and knees appropriately extended. Pause at the bottom for one second to feel the hamstring stretch. Squeeze glutes hard at lockout. Keep bar or dumbbells dragging along your legs throughout.")
        elif avg_knee < 145:
            print("  VERDICT: Too much knee bend — becoming a regular deadlift. Research shows RDL uses only about 15 degrees of knee flexion. Keep knees almost completely straight. The stretch should be entirely in your hamstrings — not your lower back.")
        else:
            print("  VERDICT: Work on hip hinge depth. Push hips further back. Lower until you feel a strong hamstring pull. If you feel it in your lower back your spine is rounding — reduce weight immediately.")

    elif detected == "LUNGE":
        print("\n  LUNGE ANALYSIS\n")
        print(f"  Depth Assessment   : ", end="")
        if min_knee <= 90:
            print("EXCELLENT — Full depth")
        elif min_knee <= 110:
            print("GOOD — Good depth")
        else:
            print("POOR — Not deep enough")

        print()
        if min_knee <= 100:
            print("  VERDICT: Good lunge depth. Front knee should track over ankle — not caving inward. Torso upright throughout. Back knee should nearly touch the ground. Train both legs equally to avoid imbalances.")
        elif min_knee <= 120:
            print("  VERDICT: Getting close to proper depth but not there yet. Take a longer stride so front thigh reaches parallel to floor. Step long, lower hips straight down, keep torso vertical.")
        else:
            print("  VERDICT: Lunge depth is insufficient. Front thigh should be parallel to floor at the bottom — that is where maximum glute and quad activation happens. Use bodyweight lunges and focus on depth before adding any load.")

    elif detected == "CALF RAISE":
        print("\n  CALF RAISE ANALYSIS\n")
        print(f"  Knee Stability     : ", end="")
        if knee_range < 8:
            print("EXCELLENT — Knees completely stable")
        elif knee_range < 15:
            print("GOOD — Minimal knee movement")
        else:
            print("MODERATE — More knee movement than ideal")

        print()
        print("  VERDICT: Go all the way up onto the balls of your feet at the top and hold for 2 seconds. Let heels drop as far below the step as possible at the bottom for the full stretch. The slow lowering phase — 3 to 4 seconds — is where most calf development happens. High reps with full range always beats heavy weight with short range.")

    elif detected == "PUSH UP":
        print("\n  PUSH UP ANALYSIS\n")
        print(f"  Depth Assessment   : ", end="")
        if min_elbow < 80:
            print("EXCELLENT — Full depth, chest near floor")
        elif min_elbow < 100:
            print("GOOD — Good depth")
        else:
            print("POOR — Not deep enough")

        print(f"  Range of Motion    : ", end="")
        if elbow_range > 70:
            print("EXCELLENT — Full range")
        elif elbow_range > 50:
            print("GOOD — Good range")
        else:
            print("POOR — Limited range")

        print()
        if min_elbow < 90 and elbow_range > 65:
            print("  VERDICT: Strong push up form. Good depth and range. Try adding a 3 second lowering phase on every rep — massively increases difficulty. Elbows at roughly 45 degrees from body — not flared wide and not fully tucked in. Body in a perfectly rigid straight line from head to heels throughout.")
        elif min_elbow >= 90:
            print("  VERDICT: Not deep enough. Chest should nearly touch the floor on every rep. Half reps build half the muscle. Drop to knee push ups if needed — full range with less load is always better than partial range with more.")
        elif elbow_range < 50:
            print("  VERDICT: Range of motion is very limited. Consciously think about touching your chest to the floor on every rep. Stretch chest and shoulders daily.")
        else:
            print("  VERDICT: Decent push up mechanics. Slow everything down, own the form, then build reps.")

    elif detected == "SHOULDER PRESS":
        print("\n  SHOULDER PRESS ANALYSIS\n")
        print(f"  Overhead Reach     : ", end="")
        if wrist_above_pct > 60:
            print("GOOD — Wrists reaching overhead consistently")
        elif wrist_above_pct > 40:
            print("MODERATE — Partial overhead range")
        else:
            print("POOR — Wrists not reaching overhead")

        print(f"  Range of Motion    : ", end="")
        if elbow_range > 60:
            print("EXCELLENT — Full range")
        elif elbow_range > 40:
            print("GOOD — Adequate range")
        else:
            print("POOR — Limited range")

        print()
        if wrist_above_pct > 55 and elbow_range > 50:
            print("  VERDICT: Good shoulder press mechanics. Fully lock elbows at top every rep. At bottom, elbows should be at 90 degrees or below. Do not arch lower back to get weight overhead — if you are arching significantly the weight is too heavy.")
        elif wrist_above_pct < 35:
            print("  VERDICT: Arms not reaching overhead. Reduce weight and press all the way until arms are fully extended on every rep. Limited overhead range often means tight lats — add lat and overhead stretching to warm-up daily.")
        else:
            print("  VERDICT: Range of motion needs work. Movement should go from shoulder height all the way to full lockout overhead. Focus on the top lockout — fully extend every rep.")

    elif detected == "BICEP CURL":
        print("\n  BICEP CURL ANALYSIS\n")
        print(f"  Range of Motion    : ", end="")
        if elbow_range > 90 and min_elbow < 55:
            print("EXCELLENT — Full flexion and extension")
        elif elbow_range > 70:
            print("GOOD — Good range")
        else:
            print("POOR — Partial reps")

        print(f"  Form Strictness    : ", end="")
        if avg_torso_vert > 0.20 and avg_shoulder < 50:
            print("GOOD — Minimal body swing")
        else:
            print("MODERATE — Some momentum detected")

        print()
        if min_elbow < 55 and elbow_range > 85:
            print("  VERDICT: Excellent bicep curl mechanics. Fully straighten arms at the bottom every rep for the complete stretch. Squeeze hard at the very top. Elbows pinned to your sides throughout — do not let them drift forward as you curl.")
        elif elbow_range < 60:
            print("  VERDICT: Range of motion too short. Reduce weight by 30 percent and go from fully straight arms at the bottom to maximum flexion at the top. Full range with less load builds more muscle than short range with heavy weight.")
        elif avg_shoulder > 60:
            print("  VERDICT: Using shoulders to help lift — body swing detected. Weight is too heavy and biceps are not doing the actual work. Reduce weight until movement completes with upper arms completely stationary.")
        else:
            print("  VERDICT: Decent curl form. Fully extend at the bottom of every rep — this stretch is where a lot of the muscle stimulus comes from.")

    elif detected == "LATERAL RAISE":
        print("\n  LATERAL RAISE ANALYSIS\n")
        print(f"  Shoulder Elevation : ", end="")
        if max_shoulder_abduct > 85:
            print("GOOD — Arms reaching shoulder height")
        elif max_shoulder_abduct > 65:
            print("MODERATE — Partially reaching shoulder height")
        else:
            print("LOW — Not raising arms high enough")

        print()
        print("  VERDICT: Raise arms until exactly parallel to floor — no higher as that shifts load to traps. Lead with elbows not wrists. Keep slight elbow bend throughout. Use lighter weight than you think you need. Slow the lowering phase to 3-4 seconds for maximum side delt development.")

    elif detected == "BENT OVER ROW":
        print("\n  BENT OVER ROW ANALYSIS\n")
        print(f"  Torso Position     : ", end="")
        if avg_torso_vert < 0.15:
            print("EXCELLENT — Good forward lean maintained")
        elif avg_torso_vert < 0.22:
            print("GOOD — Adequate torso angle")
        else:
            print("POOR — Torso too upright for effective rowing")

        print(f"  Pull Range         : ", end="")
        if elbow_range > 60:
            print("GOOD — Strong elbow pull")
        elif elbow_range > 40:
            print("MODERATE — Adequate pull")
        else:
            print("POOR — Limited pulling range")

        print()
        if avg_torso_vert < 0.20 and elbow_range > 45:
            print("  VERDICT: Good bent over row mechanics. Pull toward your lower chest and squeeze shoulder blades together hard at the top — hold for one second. Back completely flat throughout. The moment your lower back rounds under load this exercise becomes dangerous. Core braced on every single rep.")
        elif avg_torso_vert > 0.25:
            print("  VERDICT: Torso too upright. Hinge at the hip until torso is roughly 45 degrees from vertical. An upright torso row barely activates lats and turns into a bicep exercise. Hinge forward, keep back flat, then pull.")
        else:
            print("  VERDICT: Work on elbow pull range. Elbow should drive well behind your torso at the top — that is full lat range of motion. Reduce weight slightly and add lat stretching to your routine.")

    else:
        print("\n  Could not confidently detect a specific exercise.")
        print(f"  Highest score was {confidence} — below the confidence threshold of 25.")
        print()
        print("  Tips for better detection:")
        print("  - Record from SIDE ANGLE — most important factor")
        print("  - Full body visible in frame throughout")
        print("  - Good lighting — avoid backlit environments")
        print("  - Controlled pace — very fast reps reduce landmark accuracy")
        print()
        print("  Supported: squat | deadlift | rdl | lunge | calf raise")
        print("             push up | shoulder press | bicep curl | lateral raise | bent over row")

    print("\n" + "=" * 62 + "\n")


# ============================================================
#  MAIN MENU LOOP
# ============================================================

print("\n" + "=" * 62)
print("          FORMIQ — AI BIOMECHANICAL ANALYZER")
print("=" * 62)
print("\n  Supported exercises:")
print()
print("  LOWER BODY : squat | deadlift | rdl | lunge | calf raise")
print("  UPPER BODY : push up | shoulder press | bicep curl")
print("               lateral raise | bent over row")
print()
print("  Tip: Video file names can be any of these variants —")
print("       pushup / push_up / push-up / pushups")
print("       bentoverrow / bent_over_row / row")
print("       shoulderpress / shoulder_press   etc.")

while True:
    print("\n" + "-" * 62)
    user_input = input("  Enter exercise name (or 'exit' to quit): ").strip().lower()

    if user_input == "exit":
        print("\n  Session complete. Train hard, train smart.\n")
        break

    if user_input in VIDEO_ALIASES:
        aliases   = VIDEO_ALIASES[user_input]
        video_path = find_video(aliases)

        if video_path:
            print(f"\n  Found: {video_path}")
            analyze_video(video_path)
        else:
            print(f"\n  ERROR: No video found for '{user_input}'.")
            print(f"  Searched for: {', '.join(a + '.mp4' for a in aliases[:4])} etc.")
            print("  Make sure the video file is in the BIOMECH folder.")
            print("  Any of these names work: " + " | ".join(aliases[:5]))
    else:
        print("\n  Not recognized. Type one of the exercise names shown above.")