import cv2

img = cv2.imread("/opt/sam2_project/input_images2/photo10.png")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

def on_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"({x},{y})  HSV = {hsv[y, x]}")

cv2.namedWindow("img", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("img", on_click)
cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
