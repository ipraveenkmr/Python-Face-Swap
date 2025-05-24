import cv2
import dlib
import numpy as np

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

def get_landmarks(im):
    rects = detector(im, 1)
    if len(rects) != 1:
        raise Exception("Image must contain exactly one face.")
    return np.array([[p.x, p.y] for p in predictor(im, rects[0]).parts()]), rects[0]

def transformation_from_points(points1, points2):
    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)
    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2
    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2
    U, S, Vt = np.linalg.svd(points1.T @ points2)
    R = (U @ Vt).T
    return np.vstack([np.hstack(((s2 / s1) * R, (c2 - (s2 / s1) * R @ c1).reshape(2, 1))), np.array([0., 0., 1.])])

def get_face_mask(image_shape, landmarks):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    hull = cv2.convexHull(landmarks.astype(np.int32))
    cv2.fillConvexPoly(mask, hull, 255)
    return mask

def warp_image(image, M, shape):
    return cv2.warpAffine(image, M[:2], (shape[1], shape[0]), flags=cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT)

# Load images
im1 = cv2.imread("face1.jpg")  # destination face
im2 = cv2.imread("face2.jpg")  # face to be swapped in

# Get landmarks
landmarks1, rect1 = get_landmarks(im1)
landmarks2, rect2 = get_landmarks(im2)

# Get affine transform
M = transformation_from_points(landmarks2, landmarks1)

# Warp face2 to match face1
warped_im2 = warp_image(im2, M, im1.shape)

# Create mask from warped landmarks
mask = get_face_mask(im1.shape, landmarks1)

# Define center for seamless clone (center of destination face)
center = (rect1.left() + rect1.width() // 2, rect1.top() + rect1.height() // 2)

# Blend the warped face onto the target face using seamlessClone
output = cv2.seamlessClone(warped_im2, im1, mask, center, cv2.NORMAL_CLONE)

# Save output
cv2.imwrite("swapped.jpg", output)
