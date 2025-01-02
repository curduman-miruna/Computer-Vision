import cv2
import numpy as np

emoji = np.zeros((500, 500, 3), np.uint8)

cv2.rectangle(emoji, (125, 125), (375, 375), (255, 255, 255), -1)
cv2.circle(emoji, (125, 250), 125, (255, 255, 255), -1)
cv2.circle(emoji, (250, 125), 125, (255, 255, 255), -1)

# Gradient for square
for i in range(125, 375):
    for j in range(125, 375):
        emoji[i, j] = (255 - i // 2, 255 - j // 2, 255)

# Gradient for the first circle
for i in range(0, 500):
    for j in range(0, 500):
        if (i - 125) ** 2 + (j - 250) ** 2 <= 125 ** 2:
            emoji[i, j] = (255 - i // 2, 255 - j // 2, 255)

# Gradient for the second circle
for i in range(0, 500):
    for j in range(0, 500):
        if (i - 250) ** 2 + (j - 125) ** 2 <= 125 ** 2:
            emoji[i, j] = (255 - i // 2, 255 - j // 2, 255)

# Contour for heart shape
heart_mask = np.zeros_like(emoji)
cv2.rectangle(heart_mask, (125, 125), (375, 375), (255, 255, 255), -1)
cv2.circle(heart_mask, (125, 250), 125, (255, 255, 255), -1)
cv2.circle(heart_mask, (250, 125), 125, (255, 255, 255), -1)

heart_mask_gray = cv2.cvtColor(heart_mask, cv2.COLOR_BGR2GRAY)
contours, _ = cv2.findContours(heart_mask_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(emoji, contours, -1, (255, 255, 255), 3)

# Rotation of the emoji
rows, cols = emoji.shape[:2]
M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 315, 1)
rotated_emoji = cv2.warpAffine(emoji, M, (cols, rows))

# Draw sparkle with gradient
def draw_sparkle_with_gradient(img, center, size):
    pts = np.array([
        [center[0], center[1] - size * 1.2],  # Top point
        [center[0] + size // 6, center[1] - size // 6],  # Top-right diagonal
        [center[0] + size, center[1]],  # Right point
        [center[0] + size // 6, center[1] + size // 6],  # Bottom-right diagonal
        [center[0], center[1] + size * 1.2],  # Bottom point
        [center[0] - size // 6, center[1] + size // 6],  # Bottom-left diagonal
        [center[0] - size, center[1]],  # Left point
        [center[0] - size // 6, center[1] - size // 6]  # Top-left diagonal
    ], np.int32)

    pts = pts.reshape((-1, 1, 2))

    mask = np.zeros_like(img, np.uint8)
    cv2.fillPoly(mask, [pts], (255, 255, 255))
    x, y, w, h = cv2.boundingRect(pts)

    for i in range(y, y + h):
        for j in range(x, x + w):
            if cv2.pointPolygonTest(pts, (j, i), False) >= 0:
                # Calculate distance from the center
                distance = np.sqrt((j - center[0])**2 + (i - center[1])**2)
                ratio = distance / size
                color = (0, int(255 - ratio * 90), 255)
                img[i, j] = color

    mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(mask_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (255, 255, 255), 2)

draw_sparkle_with_gradient(rotated_emoji, (425, 75), 60)
draw_sparkle_with_gradient(rotated_emoji, (150, 320), 60)

rotated_emoji = cv2.copyMakeBorder(rotated_emoji, 40, 0, 40, 40, cv2.BORDER_CONSTANT, value=(0,0,0))

cv2.imshow('emoji', rotated_emoji)
cv2.imwrite('emoji.jpg', rotated_emoji)

if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()
