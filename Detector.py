import cv2
import numpy as np
import time
import math
from collections import deque


# ============================================================
# CONFIGURATION
# ============================================================

CAMERA_INDEX = 0

# Optical flow parameters
FLOW_SCALE = 0.5
MIN_FLOW_MAGNITUDE = 0.8

# Spin detection
SPIN_MAG_THRESHOLD = 1.5
SPIN_FLOW_PEAK_HOLD = 1.5

# History
FLOW_HISTORY_SECONDS = 2.0

# Display
WINDOW_NAME = "Spin / Motion Detector"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def calculate_flow_features(prev_gray, gray):
    """
    Calculate dense optical flow using Farneback algorithm.

    Returns:
        magnitude
        angle
        coherence
        score
    """

    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )

    fx = flow[..., 0]
    fy = flow[..., 1]

    magnitude, angle = cv2.cartToPolar(
        fx,
        fy,
        angleInDegrees=False
    )

    # Remove very small movements/noise
    valid = magnitude > MIN_FLOW_MAGNITUDE

    if np.count_nonzero(valid) == 0:
        return 0.0, 0.0, 0.0, 0.0, flow

    mag_values = magnitude[valid]
    angle_values = angle[valid]

    # Average motion magnitude
    avg_magnitude = float(np.mean(mag_values))

    # --------------------------------------------------------
    # Coherence
    # --------------------------------------------------------

    # Circular mean of flow direction
    mean_x = np.mean(np.cos(angle_values))
    mean_y = np.mean(np.sin(angle_values))

    coherence = math.sqrt(
        mean_x ** 2 +
        mean_y ** 2
    )

    # --------------------------------------------------------
    # Motion score
    # --------------------------------------------------------

    score = avg_magnitude * coherence

    return (
        avg_magnitude,
        angle,
        coherence,
        score,
        flow
    )


def draw_flow(frame, flow, step=20):
    """
    Draw optical flow vectors on the frame.
    """

    h, w = frame.shape[:2]

    for y in range(0, h, step):
        for x in range(0, w, step):

            fx, fy = flow[y, x]

            end_x = int(x + fx)
            end_y = int(y + fy)

            cv2.line(
                frame,
                (x, y),
                (end_x, end_y),
                (0, 255, 0),
                1
            )

            cv2.circle(
                frame,
                (x, y),
                1,
                (0, 255, 0),
                -1
            )

    return frame


# ============================================================
# SPIN DETECTOR
# ============================================================

class SpinDetector:

    def __init__(self):

        self.flow_peak_history = []

        self.flow_history = deque()

        self.last_flow_magnitude_debug = 0.0
        self.last_flow_coherence_debug = 0.0
        self.last_flow_score_debug = 0.0
        self.last_flow_peak_debug = 0.0

        self.last_time = time.time()

    def update(self, magnitude, coherence, score):

        now = time.time()

        # ----------------------------------------------------
        # Visible section from the video
        # ----------------------------------------------------

        self.flow_peak_history.append(
            (now, score)
        )

        self.flow_peak_history = [
            (t, s)
            for t, s in self.flow_peak_history
            if now - t < SPIN_FLOW_PEAK_HOLD
        ]

        self.last_flow_magnitude_debug = magnitude

        self.last_flow_coherence_debug = coherence

        self.last_flow_score_debug = score

        self.last_flow_peak_debug = max(
            (
                s
                for _, s in self.flow_peak_history
            ),
            default=0.0
        )

        # ----------------------------------------------------
        # Flow history
        # ----------------------------------------------------

        self.flow_history.append(
            (now, magnitude)
        )

        while self.flow_history:
            if now - self.flow_history[0][0] > FLOW_HISTORY_SECONDS:
                self.flow_history.popleft()
            else:
                break

        # ----------------------------------------------------
        # Elevated motion
        # ----------------------------------------------------

        elevated = sum(
            1
            for _, m in self.flow_history
            if m > SPIN_MAG_THRESHOLD
        )

        # ----------------------------------------------------
        # Spin condition
        # ----------------------------------------------------

        spin_detected = (
            self.last_flow_peak_debug > SPIN_MAG_THRESHOLD
            and
            coherence > 0.25
            and
            elevated >= 3
        )

        return spin_detected, elevated


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    detector = SpinDetector()

    prev_gray = None

    last_spin_time = 0

    print("=" * 50)
    print("OPTICAL FLOW SPIN DETECTOR")
    print("=" * 50)
    print("Press Q to quit")
    print("Press F to toggle optical flow")
    print("=" * 50)

    show_flow = False

    while True:

        ret, frame = cap.read()

        if not ret:
            print("Could not read webcam frame.")
            break

        # Flip camera like a mirror
        frame = cv2.flip(frame, 1)

        # Resize for faster processing
        small = cv2.resize(
            frame,
            None,
            fx=FLOW_SCALE,
            fy=FLOW_SCALE
        )

        gray = cv2.cvtColor(
            small,
            cv2.COLOR_BGR2GRAY
        )

        # ----------------------------------------------------
        # First frame
        # ----------------------------------------------------

        if prev_gray is None:

            prev_gray = gray.copy()

            cv2.imshow(
                WINDOW_NAME,
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            continue

        # ----------------------------------------------------
        # Optical Flow
        # ----------------------------------------------------

        (
            magnitude,
            angle,
            coherence,
            score,
            flow
        ) = calculate_flow_features(
            prev_gray,
            gray
        )

        # Update previous frame
        prev_gray = gray.copy()

        # ----------------------------------------------------
        # Detector
        # ----------------------------------------------------

        spin_detected, elevated = detector.update(
            magnitude,
            coherence,
            score
        )

        # ----------------------------------------------------
        # Prevent continuous detection
        # ----------------------------------------------------

        now = time.time()

        if spin_detected:

            if now - last_spin_time > 1.5:

                print(
                    f"[SPIN DETECTED] "
                    f"magnitude={magnitude:.2f}, "
                    f"coherence={coherence:.2f}, "
                    f"score={score:.2f}"
                )

                last_spin_time = now

        # ----------------------------------------------------
        # Draw optical flow
        # ----------------------------------------------------

        display = frame.copy()

        if show_flow:

            flow_display = cv2.resize(
                small,
                (frame.shape[1], frame.shape[0])
            )

            # Scale vectors
            flow_scaled = flow * (1 / FLOW_SCALE)

            display = draw_flow(
                display,
                flow_scaled,
                step=25
            )

        # ----------------------------------------------------
        # Status
        # ----------------------------------------------------

        if spin_detected:

            status = "SPIN / MOTION DETECTED"

        else:

            status = "NORMAL"

        cv2.putText(
            display,
            status,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0) if not spin_detected else (0, 0, 255),
            2
        )

        # ----------------------------------------------------
        # Debug information
        # ----------------------------------------------------

        cv2.putText(
            display,
            f"Flow magnitude : {magnitude:.2f}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1
        )

        cv2.putText(
            display,
            f"Coherence      : {coherence:.2f}",
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1
        )

        cv2.putText(
            display,
            f"Flow score     : {score:.2f}",
            (20, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1
        )

        cv2.putText(
            display,
            f"Peak score     : {detector.last_flow_peak_debug:.2f}",
            (20, 155),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1
        )

        cv2.putText(
            display,
            f"Elevated       : {elevated}",
            (20, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1
        )

        cv2.putText(
            display,
            "Q = Quit | F = Flow",
            (20, display.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (200, 200, 200),
            1
        )

        # ----------------------------------------------------
        # Show
        # ----------------------------------------------------

        cv2.imshow(
            WINDOW_NAME,
            display
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("f"):
            show_flow = not show_flow

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()