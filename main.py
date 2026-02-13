import cv2
import numpy as np

bandage = cv2.imread("bandage.png", cv2.IMREAD_UNCHANGED)

if bandage is None:
    print("Bandage image not found!")
    exit()

cap = cv2.VideoCapture(0)

# ✅ Define smoothing variables OUTSIDE loop
prev_cx = 0
prev_cy = 0


def overlay_bandage(frame, bandage, cx, cy, size):
    bandage_resized = cv2.resize(bandage, (size, size))
    h, w = bandage_resized.shape[:2]

    x1 = int(cx - w / 2)
    y1 = int(cy - h / 2)
    x2 = x1 + w
    y2 = y1 + h

    if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
        return frame

    alpha = bandage_resized[:, :, 3] / 255.0

    for c in range(3):
        frame[y1:y2, x1:x2, c] = (
            alpha * bandage_resized[:, :, c] +
            (1 - alpha) * frame[y1:y2, x1:x2, c]
        )

    return frame


while True:
    ret, frame = cap.read()
    if not ret:
        break

    original = frame.copy()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 100, 50])
    upper_red1 = np.array([15, 255, 255])

    lower_red2 = np.array([165, 100, 50])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2

    kernel = np.ones((9, 9), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.dilate(mask, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        if area > 2000:
            x, y, w, h = cv2.boundingRect(largest)
            cx = x + w // 2
            cy = y + h // 2

            # ✅ First frame fix
            if prev_cx == 0 and prev_cy == 0:
                prev_cx, prev_cy = cx, cy

            # ✅ Smoothing
            cx = int(0.7 * prev_cx + 0.3 * cx)
            cy = int(0.7 * prev_cy + 0.3 * cy)

            prev_cx, prev_cy = cx, cy

            size = max(w, h)

            frame = overlay_bandage(frame, bandage, cx, cy, size)

    combined = np.hstack((original, frame))
    cv2.imshow("Original | Bandage Applied", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



























# import cv2
# import numpy as np

# bandage = cv2.imread("bandage.png", cv2.IMREAD_UNCHANGED)

# if bandage is None:
#     print("Bandage image not found!")
#     exit()

# cap = cv2.VideoCapture(0)

# def overlay_bandage(frame, bandage, cx, cy, size):
#     bandage_resized = cv2.resize(bandage, (size, size))
#     h, w = bandage_resized.shape[:2]

#     x1 = int(cx - w / 2)
#     y1 = int(cy - h / 2)
#     x2 = x1 + w
#     y2 = y1 + h

#     if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
#         return frame

#     alpha = bandage_resized[:, :, 3] / 255.0

#     for c in range(3):
#         frame[y1:y2, x1:x2, c] = (
#             alpha * bandage_resized[:, :, c] +
#             (1 - alpha) * frame[y1:y2, x1:x2, c]
#         )

#     return frame
#     prev_cx, prev_cy = 0, 0

#     prev_cx = 0
#     prev_cy = 0



# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     original = frame.copy()

#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

#     # 🔥 More flexible red detection
#     lower_red1 = np.array([0, 100, 50])
#     upper_red1 = np.array([15, 255, 255])

#     lower_red2 = np.array([165, 100, 50])
#     upper_red2 = np.array([180, 255, 255])

#     mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
#     mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
#     mask = mask1 + mask2

#     # Strong cleaning
#     kernel = np.ones((9,9), np.uint8)
#     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
#     mask = cv2.dilate(mask, kernel, iterations=2)

#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     if contours:
#      largest = max(contours, key=cv2.contourArea)
#     area = cv2.contourArea(largest)

#     if area > 2000:
#         x, y, w, h = cv2.boundingRect(largest)
#         cx = x + w // 2
#         cy = y + h // 2

#         # 🔥 Smoothing (reduce shake)
#         cx = int(0.7 * prev_cx + 0.3 * cx)
#         cy = int(0.7 * prev_cy + 0.3 * cy)

#         prev_cx, prev_cy = cx, cy

#         size = max(w, h)

#         frame = overlay_bandage(frame, bandage, cx, cy, size)


#     combined = np.hstack((original, frame))
#     cv2.imshow("Original | Bandage Applied", combined)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
