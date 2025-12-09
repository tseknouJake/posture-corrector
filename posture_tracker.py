import cv2
import math
import time
from collections import deque
import mediapipe as mp
from pync import Notifier

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def angle_from_vertical(p_bottom, p_top):
    """
    Return angle in degrees between the vector (bottom->top)
    and the vertical axis. 0° = perfectly vertical; larger = more tilted.
    p_bottom, p_top: (x, y) in normalized or pixel coordinates.
    """
    vx = p_top[0] - p_bottom[0]
    vy = p_top[1] - p_bottom[1]

    dot = vx * 0 + vy * (-1)
    mag_v = math.sqrt(vx * vx + vy * vy)
    if mag_v == 0:
        return 0.0

    cos_theta = dot / mag_v
    cos_theta = max(-1.0, min(1.0, cos_theta))
    theta = math.degrees(math.acos(cos_theta))
    return theta 


def compute_posture_metrics(landmarks):
    """
    landmarks: list of pose landmarks in normalized coordinates [0..1].
    Returns a dict with simple posture metrics.
    """

    def get(lm_enum):
        lm = landmarks[lm_enum.value]
        return lm.x, lm.y, lm.visibility

    ls_x, ls_y, ls_vis = get(mp_pose.PoseLandmark.LEFT_SHOULDER)
    rs_x, rs_y, rs_vis = get(mp_pose.PoseLandmark.RIGHT_SHOULDER)
    lh_x, lh_y, lh_vis = get(mp_pose.PoseLandmark.LEFT_HIP)
    rh_x, rh_y, rh_vis = get(mp_pose.PoseLandmark.RIGHT_HIP)
    le_x, le_y, le_vis = get(mp_pose.PoseLandmark.LEFT_EAR)
    re_x, re_y, re_vis = get(mp_pose.PoseLandmark.RIGHT_EAR)

    mid_shoulder = ((ls_x + rs_x) / 2.0, (ls_y + rs_y) / 2.0)
    mid_hip = ((lh_x + rh_x) / 2.0, (lh_y + rh_y) / 2.0)

    if le_vis >= re_vis:
        head_point = (le_x, le_y)
    else:
        head_point = (re_x, re_y)

    head_angle = angle_from_vertical(mid_shoulder, head_point)

    torso_angle = angle_from_vertical(mid_hip, mid_shoulder)

    shoulder_tilt = (ls_y - rs_y) * 100.0

    return {
        "head_angle_deg": head_angle,
        "torso_angle_deg": torso_angle,
        "shoulder_tilt": shoulder_tilt,
    }


def classify_posture(metrics, baseline=None):
    """
    baseline: dict of same keys as metrics, representing calibrated "good posture".
    If baseline is given, classify based on deviation from baseline.
    """
    if baseline is not None:
        head = metrics["head_angle_deg"] - baseline["head_angle_deg"]
        torso = metrics["torso_angle_deg"] - baseline["torso_angle_deg"]
        tilt = abs(metrics["shoulder_tilt"] - baseline["shoulder_tilt"])
    else:
        head = metrics["head_angle_deg"]
        torso = metrics["torso_angle_deg"]
        tilt = abs(metrics["shoulder_tilt"])

    bad_reasons = []


    if head > 13:
        bad_reasons.append("forward head")
    if torso > 12:
        bad_reasons.append("leaning torso")
    if tilt > 6:  
        bad_reasons.append("shoulder tilt")

    if bad_reasons:
        return "BAD", bad_reasons
    else:
        return "GOOD", []


def smooth_value(history: deque, new_value: float, max_len: int = 3) -> float:
    """
    Simple moving average smoother over the last max_len samples.
    Default window is 3 for faster reaction.
    """
    history.append(new_value)
    if len(history) > max_len:
        history.popleft()
    return sum(history) / len(history)


def send_posture_notification():
    Notifier.notify(
        "Your posture looks bad. Sit up straight :)",
        title="Posture Reminder",
    )


def main():
    cap = cv2.VideoCapture(0)
    alert_img = cv2.imread("myImage.png")
    if alert_img is None:
        print("WARNING: could not load myImage.png")

    alert_visible = False

    head_hist = deque()
    torso_hist = deque()
    tilt_hist = deque()

    baseline = None              
    bad_counter = 0              
    BAD_FRAMES_THRESHOLD = 30     

    last_display_label = "GOOD"
    last_notify_time = 0.0
    NOTIFY_COOLDOWN = 0.0

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = pose.process(rgb)

            posture_label_raw = "NO BODY"
            display_label = "NO BODY"
            bad_reasons = []
            metrics = None

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                )

                metrics = compute_posture_metrics(results.pose_landmarks.landmark)

                metrics["head_angle_deg"] = smooth_value(
                    head_hist, metrics["head_angle_deg"], max_len=3
                )
                metrics["torso_angle_deg"] = smooth_value(
                    torso_hist, metrics["torso_angle_deg"], max_len=3
                )
                metrics["shoulder_tilt"] = smooth_value(
                    tilt_hist, metrics["shoulder_tilt"], max_len=3
                )

                posture_label_raw, bad_reasons = classify_posture(metrics, baseline)

                if posture_label_raw == "BAD":
                    bad_counter += 1
                else:
                    bad_counter = 0 

                if bad_counter >= BAD_FRAMES_THRESHOLD:
                    display_label = "BAD"

                else:
                    display_label = "GOOD"


                text_lines = [
                    f"Head angle:  {metrics['head_angle_deg']:.1f} deg",
                    f"Torso angle: {metrics['torso_angle_deg']:.1f} deg",
                    f"Shoulder tilt: {metrics['shoulder_tilt']:.1f}",
                    f"Baseline: {'YES' if baseline is not None else 'NO'}",
                    "Press 'c' to calibrate good posture",
                ]
                y0 = 20
                for line in text_lines:
                    cv2.putText(
                        frame,
                        line,
                        (10, y0),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
                    y0 += 20

            now = time.time()
            if (
                display_label == "BAD"
                and last_display_label != "BAD"
                and (now - last_notify_time) > NOTIFY_COOLDOWN
            ):
                send_posture_notification()
                last_notify_time = now

            last_display_label = display_label

            color = (0, 255, 0) if display_label == "GOOD" else (0, 0, 255)
            cv2.putText(
                frame,
                f"Posture: {display_label}",
                (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
                cv2.LINE_AA,
            )

            if display_label == "BAD" and bad_reasons:
                reason_text = ", ".join(bad_reasons)
                cv2.putText(
                    frame,
                    reason_text,
                    (10, h - 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow("Posture Tracker (prototype)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC to quit
                break
            elif key == ord('c'):

                if metrics is not None:
                    baseline = metrics.copy()
                    bad_counter = 0
                    print("Calibrated baseline:", baseline)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
