import cv2
import numpy as np


# CONFIGURES
FPS_LIM = 60
UPDATE_IF_LESS_THAN = 50

SAVE_RESULT = False
RESULT_FPS = 30


# FUNCTIONS
def get_points_to_track(frame):
    kp1, des1 = orb.detectAndCompute(marker, None)
    kp2, des2 = orb.detectAndCompute(frame, None)
    # Find matches
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)[:100]
    # Points to track
    src_pts = np.array([kp1[m.queryIdx].pt for m in matches], dtype=np.float32).reshape(-1, 1, 2)
    dst_pts = np.array([kp2[m.trainIdx].pt for m in matches], dtype=np.float32).reshape(-1, 1, 2)
    return src_pts, dst_pts


# Read image
marker = cv2.imread("Task3/data/marker.jpg", cv2.IMREAD_GRAYSCALE)

# Read video
video = cv2.VideoCapture("Task3/data/find_chocolate.mp4")
if not video.isOpened():
    print("Error: unable to open video!")
    exit(0)
ret, old_frame = video.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# Def marker borders
border = np.float32([
    [0, 0],
    [0, marker.shape[0] - 1],
    [marker.shape[1] - 1, marker.shape[0] - 1],
    [marker.shape[1] - 1, 0]
    ]).reshape(-1, 1, 2)

# ORB & Brute-Force Matcher
orb = cv2.ORB_create(WTA_K=3, nlevels=18)
bf = cv2.BFMatcher(cv2.NORM_HAMMING2, crossCheck=True)

# Finding keypoints to track
src, dst = get_points_to_track(old_gray)

# Parameters for lucas kanade optical flow
lk_params = dict(
    winSize = (31, 31),
    maxLevel = 3,
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

if SAVE_RESULT:
    result = cv2.VideoWriter("Task3/result/result2.mp4", -1,
    RESULT_FPS, (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))))

while(True):
    # Find homography and apply
    M, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    transformed_border = cv2.perspectiveTransform(border, M).astype(np.int32)

    # Draw old frame with transformed tracker
    display = cv2.polylines(old_gray, [transformed_border], True, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow("Tracker initialization", display)

    # Read next frame
    ret, frame = video.read()
    if not ret:
        break
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate optical flow
    dst, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, dst, None, **lk_params)
    
    # Select good points, save to T-1
    old_gray = frame_gray.copy()
    dst = dst[st==1].reshape(-1, 1, 2)
    src = src[st==1].reshape(-1, 1, 2)
    print(src.shape[0])

    # Get new points if need
    if dst.shape[0] < UPDATE_IF_LESS_THAN:
        src, dst = get_points_to_track(old_gray)

    # Save if specified
    if SAVE_RESULT:
        result.write(frame)
    
    if cv2.waitKey(1000 // FPS_LIM) == ord("q"):
        break

video.release()
cv2.destroyAllWindows()