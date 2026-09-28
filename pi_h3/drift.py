"""Post-edit pixel-drift alignment, adapted from Mozer's ComfyUI-PixelDriftFix (flat_4_points).

Realigns an edited image to its source so framing drift, stretching and minor
cropping from the edit are removed. Falls back to the untouched edit when fewer
than ten reliable feature matches exist. Requires only OpenCV + NumPy.
"""
import numpy as np


def align(source, edited, min_matches=10):
    """Return a source-aligned PIL image, or None when alignment is not possible."""
    import cv2
    source_cv = cv2.cvtColor(np.array(source.convert('RGB')), cv2.COLOR_RGB2BGR)
    edited_cv = cv2.cvtColor(np.array(edited.convert('RGB')), cv2.COLOR_RGB2BGR)
    height, width = source_cv.shape[:2]
    gray_source = cv2.cvtColor(source_cv, cv2.COLOR_BGR2GRAY)
    gray_edited = cv2.cvtColor(edited_cv, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(nfeatures=0, nOctaveLayers=3, contrastThreshold=0.01, edgeThreshold=20, sigma=1.6)
    kp_source, des_source = sift.detectAndCompute(gray_source, None)
    kp_edited, des_edited = sift.detectAndCompute(gray_edited, None)
    if des_source is None or des_edited is None:
        return None
    matches = cv2.BFMatcher().knnMatch(des_edited, des_source, k=2)
    good = [m for pair in matches if len(pair) == 2 for m, n in (pair,) if m.distance < 0.80 * n.distance]
    if len(good) < min_matches:
        return None
    points_edited = np.float32([kp_edited[m.queryIdx].pt for m in good])
    points_source = np.float32([kp_source[m.trainIdx].pt for m in good])
    matrix, mask = cv2.findHomography(points_edited, points_source, cv2.RANSAC, 5.0)
    if matrix is None or mask is None or int(mask.sum()) < min_matches:
        return None
    warped = cv2.warpPerspective(edited_cv, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)
    from PIL import Image
    return Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))