from tensorflow.keras.models import load_model
import cv2
import numpy as np
import joblib
from imutils.contours import sort_contours

def preprocessing(img):
   blurred_image = blur(img)
   thresh_binary = thresholding(blurred_image, 180, cv2.THRESH_BINARY)
   thresh_image = thresh_binary
   grayscale_thresh = grayscale(thresh_image)
   edges_image = edge_detection(grayscale_thresh, "canny")
   kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 5)) #kernel, iterasi, kernel size
   dilasi_image= dilasi(edges_image, kernel_rect, 3)
   rect = find_contour(dilasi_image)
   crop_image = transform_perspective(img, rect.reshape(4, 2))
   
   img = crop_image
   
   return img
   
   
   
   
   
   
   
   
   
def predict(img):
    model = load_model('models/model.h5')
    labels = joblib.load('models/EduScan.pkl')

    img = preprocessing(img)
    
    pred = model.predict(img)
    pred_conf = np.amax(pred)
    pred_index = np.argmax(pred)
    pred_label = labels[pred_index]
    
    return pred_label, pred_conf




def blur(image, kernel_size=(5, 5)):
  blurred_image = cv2.GaussianBlur(image, kernel_size, 0)

  return blurred_image

def thresholding(image, threshold=180, types=cv2.THRESH_BINARY):
  _, mask = cv2.threshold(image, threshold, 225 , types)

  return mask

def grayscale(image):

  grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  return grayscale_image

def edge_detection(image, edge_detection_method="canny"):

  if edge_detection_method == "canny":
    edges_image = cv2.Canny(image, 50, 150)
  elif edge_detection_method == "sobel":
    edges_image = cv2.Sobel(image, -1, 1, 0, ksize=3)
  elif edge_detection_method == "scharr":
    edges_image = cv2.Scharr(image, -1, 1, 0)
  else:
    raise Exception("Metode edge detection tidak dikenali.")
  return edges_image

def dilasi(image, kernel, iterations=1):
    image = np.uint8(image)
    dilated_image = cv2.dilate(image, kernel, iterations)
    return dilated_image

def find_contour(edges):
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    max_area = 0
    best_rect = None
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4:
            area = cv2.contourArea(approx)
            if area > max_area:
                max_area = area
                best_rect = approx

    return best_rect

def transform_perspective(image, pts):
    rect = sort_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))

    maxWidth = max(int(widthA), int(widthB))
    maxHeight = max(int(heightA), int(heightB))

    ratio = 85.6 / 53.98

    if maxWidth > maxHeight:
        maxHeight = int(maxWidth / ratio)
    else:
        maxWidth = int(maxHeight * ratio)

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def sort_points(pts):
    if len(pts) == 4:
        rectangle = np.zeros((4, 2), dtype="float32")
        total = pts.sum(axis=1)
        rectangle[0] = pts[np.argmin(total)]
        rectangle[2] = pts[np.argmax(total)]

        difference = np.diff(pts, axis=1)
        rectangle[1] = pts[np.argmin(difference)]
        rectangle[3] = pts[np.argmax(difference)]

        return rectangle
    else:
        raise ValueError("Input should have exactly 4 points.")
    
    
def detect_rows(image, draw_image, min_w=100, max_w=1500, min_h=8, max_h=500, show_image=True):

  detect_rows = draw_image.copy()

  contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  contours = sort_contours(contours, method="top-to-bottom")[0]

  valid_contours = []
  for contour in contours:
      x, y, w, h = cv2.boundingRect(contour)
      if(w >= min_w and w <= max_w) and (h >= min_h and h <= max_h):
        cv2.rectangle(detect_rows, (x, y), (x + w, y + h), (0, 255, 0), 7)
        valid_contours.append(contour)

  return valid_contours