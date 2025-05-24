import cv2
import dlib
import numpy as np

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"

# Initialize dlib's face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

def get_landmarks(im):
    rects = detector(im, 1)
    if len(rects) != 1:
        raise Exception("Image must contain exactly one face.")
    return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])

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
    U, S, Vt = np.linalg.svd(points1.T * points2)
    R = (U * Vt).T
    return np.vstack([np.hstack(((s2 / s1) * R, c2.T - (s2 / s1) * R * c1.T)), np.matrix([0., 0., 1.])])

def warp_im(im, M, dshape):
    output_im = np.zeros(dshape, dtype=im.dtype)
    cv2.warpAffine(im, M[:2], (dshape[1], dshape[0]), dst=output_im, borderMode=cv2.BORDER_TRANSPARENT, flags=cv2.WARP_INVERSE_MAP)
    return output_im

def correct_colours(im1, im2, landmarks1):
    blur_amount = 0.6 * np.linalg.norm(np.mean(landmarks1[36:42], axis=0) - np.mean(landmarks1[42:48], axis=0))
    blur_amount = int(blur_amount)
    if blur_amount % 2 == 0: blur_amount += 1
    im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
    im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)
    return np.clip(im2.astype(np.float64) + (im1.astype(np.float64) - im1_blur.astype(np.float64)), 0, 255).astype(np.uint8)

def get_face_mask(im, landmarks):
    im = np.zeros(im.shape[:2], dtype=np.float64)
    points = cv2.convexHull(landmarks)
    cv2.fillConvexPoly(im, points, color=1)
    im = np.array([im, im, im]).transpose((1, 2, 0))
    return im

# Load source and destination images
im1 = cv2.imread("face1.jpg")
im2 = cv2.imread("face2.jpg")
landmarks1 = get_landmarks(im1)
landmarks2 = get_landmarks(im2)

M = transformation_from_points(landmarks1, landmarks2)
mask = get_face_mask(im2, landmarks2)
warped_mask = warp_im(mask, M, im1.shape)
combined_mask = np.max([get_face_mask(im1, landmarks1), warped_mask], axis=0)

warped_im2 = warp_im(im2, M, im1.shape)
warped_corrected_im2 = correct_colours(im1, warped_im2, landmarks1)

output_im = im1 * (1.0 - combined_mask) + warped_corrected_im2 * combined_mask
cv2.imwrite("swapped.jpg", output_im.astype(np.uint8))
