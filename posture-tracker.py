import cv2
import math
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def angle_from_vertical(p_bottom, p_top):
    """
    Return angle in degrees between the vector (bottom->top)
    and the vertical axis. 0° = perfectly vertical; larger = more tilted.
    p_bottom, p_top: (x, y) in image coordinates (normalized or pixels).
    """
    vx = p_top[0] - p_bottom[0]
    vy = p_top[1] - p_bottom[1]

    # In image coords, y grows downward, so vertical "up" is (0, -1)
    dot = vx * 0 + vy * (-1)
    mag_v = math.sqrt(vx * vx + vy * vy)
    if mag_v == 0:
        return 0.0

    cos_theta = dot / mag_v  # |(0,-1)| = 1
    cos_theta = max(-1.0, min(1.0, cos_theta))  # clamp numerical errors
    theta = math.degrees(math.acos(cos_theta))
    return theta  # 0 = straight up, >0 = leaning


def compute_posture_metrics(landmarks):
    """
    landmarks: list of pose landmarks in normalized coordinates [0..1].
    Returns a dict with simple posture metrics.
    """

    # Helper to get (x,y) quickly
    def p(lm_enum):
        lm = landmarks[lm_enum.value]
        return (lm.x, lm.y)

    left_shoulder = p(mp_pose.PoseLandmark.LEFT_SHOULDER)
    right_shoulder = p(mp_pose.PoseLandmark.RIGHT_SHOULDER)
    left_hip = p(mp_pose.PoseLandmark.LEFT_HIP)
    right_hip = p(mp_pose.PoseLandmark.RIGHT_HIP)
    left_ear = p(mp_pose.PoseLandmark.LEFT_EAR)
    right_ear = p(mp_pose.PoseLandmark.RIGHT_EAR)

    # Use the side where the ear is more confidently visible.
    # For a first pass just take left.
    shoulder = left_shoulder
    ear = left_ear
    hip = left_hip

    # 1) Forward head posture: angle shoulder->ear vs vertical
    head_angle = angle_from_vertical(shoulder, ear)

    # 2) Torso lean: angle hip->shoulder vs vertical
    torso_angle = angle_from_vertical(hip, shoulder)

    # 3) Shoulder tilt: vertical difference between shoulders
    # Positive if left shoulder is lower than right (since y is downward).
    shoulder_tilt = (left_shoulder[1] - right_shoulder[1]) * 100.0  # scaled for readability

    return {
        "head_angle_deg": head_angle,
        "torso_angle_deg": torso_angle,
        "shoulder_tilt": shoulder_tilt,
    }


def classify_posture(metrics):
    """
    Very rough heuristic.
    Tune thresholds after testing and/or add calibration.
    """
    head = metrics["head_angle_deg"]
    torso = metrics["torso_angle_deg"]
    tilt = abs(metrics["shoulder_tilt"])

    bad_reasons = []

    # Example thresholds (you should tune these):
    if head > 20:        # forward head
        bad_reasons.append("forward head")
    if torso > 15:       # leaning torso
        bad_reasons.append("leaning torso")
    if tilt > 5:         # shoulders uneven
        bad_reasons.append("shoulder tilt")

    if bad_reasons:
        return "BAD", bad_reasons
    else:
        return "GOOD", []


def main():
    cap = cv2.VideoCapture(0)

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

            # Flip for mirror view
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            # Convert to RGB for MediaPipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = pose.process(rgb)

            posture_label = "NO BODY"
            bad_reasons = []

            if results.pose_landmarks:
                # Draw landmarks for debugging
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                )

                metrics = compute_posture_metrics(results.pose_landmarks.landmark)
                posture_label, bad_reasons = classify_posture(metrics)

                # Show some numeric info in the corner
                text_lines = [
                    f"Head angle:  {metrics['head_angle_deg']:.1f} deg",
                    f"Torso angle: {metrics['torso_angle_deg']:.1f} deg",
                    f"Shoulder tilt: {metrics['shoulder_tilt']:.1f}",
                ]
                y0 = 20
                for line in text_lines:
                    cv2.putText(frame, line, (10, y0), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (255, 255, 255), 1, cv2.LINE_AA)
                    y0 += 20

            # Show overall posture label at top
            color = (0, 255, 0) if posture_label == "GOOD" else (0, 0, 255)
            cv2.putText(frame, f"Posture: {posture_label}", (10, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

            # Optionally show reasons
            if bad_reasons:
                reason_text = ", ".join(bad_reasons)
                cv2.putText(frame, reason_text, (10, h - 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

            cv2.imshow("Posture Tracker (prototype)", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()