import cv2
import mediapipe as mp
import numpy as np
import json
import socket
from collections import deque
import time

# ===============================
# UDP 配置
# ===============================
UDP_IP = "127.0.0.1"
UDP_PORT = 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# ===============================
# 参数配置
# ===============================
TRAIL_LEN = 12
MOVE_THRESHOLD = 0.12

PINCH_DISTANCE_THRESHOLD = 35
PINCH_OPEN_THRESHOLD = 55
FIST_HOLD_FRAMES = 8

# Hue Bar：改为相对画面尺寸，避免 Mac / Windows 分辨率差异导致可用区间过窄
HUEBAR_X_RATIO = 0.15
HUEBAR_Y_RATIO = 0.88
HUEBAR_W_RATIO = 0.70
HUEBAR_H_RATIO = 0.03

# confirm 冷却：避免三连 pinch 一直连发
CONFIRM_COOLDOWN_SEC = 1.0

# 摄像头目标分辨率：尽量统一 Win / Mac 体验
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

# ===============================
# MediaPipe 初始化
# ===============================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5
)

# ===============================
# 手部关键点绘制样式
# ===============================
mp_draw = mp.solutions.drawing_utils

HAND_LANDMARK_STYLE = mp_draw.DrawingSpec(
    color=(220, 220, 220),
    thickness=3,
    circle_radius=2
)

HAND_CONNECTION_STYLE = mp_draw.DrawingSpec(
    color=(190, 190, 190),
    thickness=2
)

# ===============================
# 工具函数
# ===============================
def clamp01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))

def landmarks_to_np(landmarks, w, h):
    return np.array([(int(lm.x * w), int(lm.y * h)) for lm in landmarks.landmark])

def palm_center(pts):
    return np.mean(pts[[0, 1, 5, 9, 13, 17]], axis=0).astype(int)

def detect_page_flip(trail, frame_w, frame_h):
    if len(trail) < TRAIL_LEN:
        return None
    dx = (trail[-1][0] - trail[0][0]) / frame_w
    dy = (trail[-1][1] - trail[0][1]) / frame_h

    if dx > MOVE_THRESHOLD:
        return "Right"
    elif dx < -MOVE_THRESHOLD:
        return "Left"
    elif dy < -MOVE_THRESHOLD:
        return "Up"
    elif dy > MOVE_THRESHOLD:
        return "Down"
    return None

def update_direction(label, center, w, h, trail_map, hand_direction):
    trail_map[label].append(center)
    if len(trail_map[label]) >= TRAIL_LEN:
        d = detect_page_flip(trail_map[label], w, h)
        if d:
            hand_direction[label] = d
            trail_map[label].clear()
        else:
            hand_direction[label] = "Still"

def draw_text_box(img, text, org, font, scale=0.6,
                  text_color=(255, 255, 255),
                  bg_color=(0, 0, 0),
                  thickness=2):
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = org
    cv2.rectangle(
        img,
        (x - 4, y - th - 6),
        (x + tw + 4, y + baseline + 4),
        bg_color,
        -1
    )
    cv2.putText(img, text, (x, y), font, scale, text_color, thickness, cv2.LINE_AA)

# ===============================
# 手势识别
# ===============================
fist_hold_counter = {"Left": 0, "Right": 0}
pinch_phase = {"Left": "Open", "Right": "Open"}

def is_open_palm(pts):
    tips = [8, 12, 16, 20]
    mcps = [5, 9, 13, 17]
    d = [np.linalg.norm(pts[t] - pts[m]) for t, m in zip(tips, mcps)]
    return np.mean(d) > 50

def is_fist(pts, label):
    tips = [8, 12, 16, 20]
    mcps = [5, 9, 13, 17]
    curled = sum(pts[t][1] > pts[m][1] + 25 for t, m in zip(tips, mcps))

    if curled >= 3:
        fist_hold_counter[label] += 1
    else:
        fist_hold_counter[label] = 0

    return fist_hold_counter[label] >= FIST_HOLD_FRAMES

def is_pinch(pts):
    if pts is None:
        return False
    d = np.linalg.norm(pts[4] - pts[8])
    return d < PINCH_DISTANCE_THRESHOLD

# ===============================
# 主程序
# ===============================
def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    if not cap.isOpened():
        raise RuntimeError(
            "Cannot open camera. Please check camera permission in PyCharm "
            "and whether another app is using the camera."
        )

    font = cv2.FONT_HERSHEY_SIMPLEX

    trail_map = {
        "Left": deque(maxlen=TRAIL_LEN),
        "Right": deque(maxlen=TRAIL_LEN)
    }
    hand_direction = {"Left": "Still", "Right": "Still"}

    # HSV 状态（持续发送）
    hue01 = 0.0
    sat01 = 1.0
    val01 = 1.0

    slider_locked = False

    # 右手三连 pinch -> confirm_step
    pinch_count = 0
    fist_active = {"Right": False}
    last_confirm_time = -999.0

    # 调试节流
    last_print = 0.0

    while True:
        last_action = "none"

        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # 动态 Hue bar：根据当前画面尺寸计算
        bar_x = int(w * HUEBAR_X_RATIO)
        bar_y = int(h * HUEBAR_Y_RATIO)
        bar_w = int(w * HUEBAR_W_RATIO)
        bar_h = max(16, int(h * HUEBAR_H_RATIO))

        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        left_pts = None
        right_pts = None
        left_center = None

        if results.multi_hand_landmarks and results.multi_handedness:
            for lm, hd in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = hd.classification[0].label  # "Left" / "Right"
                pts = landmarks_to_np(lm, w, h)
                center = palm_center(pts)

                # 绘制关键点 + 骨架
                mp_draw.draw_landmarks(
                    frame,
                    lm,
                    mp_hands.HAND_CONNECTIONS,
                    HAND_LANDMARK_STYLE,
                    HAND_CONNECTION_STYLE
                )

                if label == "Left":
                    left_pts = pts
                    left_center = center
                elif label == "Right":
                    right_pts = pts

                update_direction(label, center, w, h, trail_map, hand_direction)

        # ======================================================
        # 1) SV 更新：左手掌心 -> S/V
        #    x -> S, y -> V(上亮下暗)
        # ======================================================
        if left_center is not None:
            nx = left_center[0] / w
            ny = left_center[1] / h
            sat01 = clamp01(nx)
            val01 = clamp01(1.0 - ny)

        # ======================================================
        # 2) Hue 更新：左手 pinch 时，用 index x 映射 hue
        #    改成动态 bar 范围，不再写死 100~500 像素
        # ======================================================
        left_is_pinch = (left_pts is not None) and is_pinch(left_pts)
        if left_is_pinch and not slider_locked:
            x = left_pts[8][0]
            hue01 = clamp01((x - bar_x) / float(bar_w))

        # ======================================================
        # 3) 右手：三连 pinch -> confirm_step（带冷却）
        # ======================================================
        if right_pts is not None:
            d = np.linalg.norm(right_pts[4] - right_pts[8])

            if pinch_phase["Right"] == "Open" and d < PINCH_DISTANCE_THRESHOLD:
                pinch_phase["Right"] = "Close"
                pinch_count += 1
            elif pinch_phase["Right"] == "Close" and d > PINCH_OPEN_THRESHOLD:
                pinch_phase["Right"] = "Open"

            now = time.time()
            if pinch_count >= 3:
                pinch_count = 0
                if (now - last_confirm_time) >= CONFIRM_COOLDOWN_SEC:
                    last_confirm_time = now
                    last_action = "confirm_step"

        # ======================================================
        # 4) 右手 Fist：cancel_step
        # ======================================================
        if right_pts is not None and is_fist(right_pts, "Right"):
            if not fist_active["Right"]:
                fist_active["Right"] = True
                last_action = "cancel_step"
        else:
            fist_active["Right"] = False

        # ======================================================
        # 5) 输出手势
        # ======================================================
        left_gesture = "none"
        right_gesture = "none"

        if left_pts is not None:
            if is_fist(left_pts, "Left"):
                left_gesture = "fist"
            elif is_pinch(left_pts):
                left_gesture = "grab"
            elif is_open_palm(left_pts):
                left_gesture = "open_palm"
            else:
                left_gesture = "other"

        if right_pts is not None:
            if is_fist(right_pts, "Right"):
                right_gesture = "fist"
            elif is_pinch(right_pts):
                right_gesture = "pinch"
            elif is_open_palm(right_pts):
                right_gesture = "open_palm"
            else:
                right_gesture = "other"

        # ======================================================
        # 6) UDP 数据
        # ======================================================
        control01 = val01
        control_locked = False

        udp_data = {
            "Left": {
                "direction": hand_direction["Left"],
                "gesture": left_gesture,
                "sv": [round(sat01, 3), round(val01, 3)]
            },
            "Right": {
                "direction": hand_direction["Right"],
                "gesture": right_gesture
            },
            "slider": {
                "value": round(hue01, 3),
                "locked": slider_locked
            },
            "controlSlider": {
                "value": round(control01, 3),
                "locked": control_locked
            },
            "action": last_action
        }

        sock.sendto(json.dumps(udp_data).encode("utf-8"), (UDP_IP, UDP_PORT))

        # 1Hz 调试打印
        now = time.time()
        if now - last_print > 1.0:
            last_print = now
            print(
                f"[UDP] action={last_action} "
                f"H={hue01:.2f} S={sat01:.2f} V={val01:.2f} "
                f"L={left_gesture} R={right_gesture}"
            )

        # ======================================================
        # GUI（调试）
        # ======================================================
        draw_text_box(frame, f"Action: {last_action}", (10, 25), font, 0.7)
        draw_text_box(frame, f"Hue: {hue01:.2f}  Sat: {sat01:.2f}  Val: {val01:.2f}", (10, 55), font, 0.6)
        draw_text_box(frame, f"Left:  {left_gesture} | Dir: {hand_direction['Left']}", (10, 85), font, 0.55)
        draw_text_box(frame, f"Right: {right_gesture} | Dir: {hand_direction['Right']}", (10, 110), font, 0.55)

        # Hue bar
        cv2.rectangle(
            frame,
            (bar_x, bar_y),
            (bar_x + bar_w, bar_y + bar_h),
            (255, 255, 255),
            2
        )
        cv2.rectangle(
            frame,
            (bar_x, bar_y),
            (bar_x + int(hue01 * bar_w), bar_y + bar_h),
            (0, 180, 255),
            -1
        )

        cv2.imshow("Hand Tracking - V2 HSV Only (Stage-free)", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()