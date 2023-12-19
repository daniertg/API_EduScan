from tensorflow.keras.models import load_model
import cv2
import numpy as np
import joblib
from imutils.contours import sort_contours
import imutils
from PIL import Image, ImageOps

model = load_model('models/model.h5')
char_list = joblib.load('models/EduScan.pkl')

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
   brigthness_img = brightness(crop_image, 1, 50)
   filter_img = red_filter(brigthness_img)
   grayscale_img = grayscale(filter_img)
   thresh_img = thresholding(grayscale_img, 180, cv2.THRESH_TRUNC)
   invers_img = 255 - grayscale_img

   kernel_custom = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 1))
   dilated_img = dilasi(invers_img, kernel_custom, 1)

   kernel_bilateral = np.array([[0.0625, 0.125, 0.0625], [0.125, 0.25, 0.125], [0.0625, 0.125, 0.0625]])
   image_enhanced = cv2.filter2D(dilated_img, -1, kernel_bilateral)

   thresh2_img = thresholding(image_enhanced, 100,cv2.THRESH_OTSU)

   valid_contours = detect_rows(thresh2_img, crop_image)
   word_rois = extract_word_rois(grayscale_img, valid_contours)
   
   return word_rois
   
   
   
def predict(img):
    global model, char_list
    
    img = preprocessing(img)
    
    pred_labels = []
    pred_confs = []
    
    for roi in img:
        pred = model.predict(roi)
        pred_conf = np.amax(pred)
        pred_index = np.argmax(pred)
        pred_label = char_list[pred_index]
        
        pred_labels.append(pred_label)
        pred_confs.append(pred_conf)
    
    return pred_labels, pred_confs




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

def brightness(image, alpha, beta):
  brigthness_img = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
  return brigthness_img

def red_filter (image) :
  image[:, :, 1] = 0
  image[:, :, 2] = 0
  return image

def extract_word_rois(grayscale_image, valid_contours):

  word_rois = []
  for word in valid_contours:
    x, y, w, h = cv2.boundingRect(word)
    word_roi = grayscale_image[y:y + h, x:x + w]
    word_rois.append(word_roi)

  return word_rois

def resize_img(img, w, h):
  if w > h:
    resized = imutils.resize(img, width=28)
  else:
    resized = imutils.resize(img, height=28)

  # (w, h) = resized.shape
  (h, w) = resized.shape

  # Calculate how many pixels need to fill char image
  dX = int(max(0, 28 - w) / 2.0)
  dY = int(max(0, 28 - h) / 2.0)

  filled = cv2.copyMakeBorder(resized, top=dY, bottom=dY, right=dX, left=dX, borderType=cv2.BORDER_CONSTANT, value=(0,0,0))
  filled = cv2.resize(filled, (28,28))

  return filled

def segment_and_recognize_letters(img):
  word_rois = preprocessing(img)
  segmented_letters = []

  for roi in word_rois:
    letters = []
    # Thresholding
    thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Opening Morphology
    kernel = np.ones((3, 3), np.uint8)
    opened_image = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, 1)


    # Localitation
    contours_letter, _ = cv2.findContours(opened_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  
    contours_letter = sort_contours(contours_letter, method="left-to-right")[0]


    lps = 0

    for contour in contours_letter:
        pred_index = 0
        # Hitung bounding box kontur
        x1, y1, w1, h1 = cv2.boundingRect(contour)

        if cv2.contourArea(contour) > 6 and w1/h1 < 2:

              letter = opened_image[y1:y1+h1, x1:x1+w1]
              #cv2_imshow(opened_image[y1:y1+h1, x1:x1+w1])
              (h, w) = letter.shape
              letter = resize_img(letter, w, h)
              #print(letter.shape)
              # Convert the image to PIL format
              pil_img = Image.fromarray(letter)
              # Tentukan ukuran padding (misalnya, 5 piksel)
              padding_size = 8

              # Buat border padding menggunakan ImageOps.expand
              padded_letter = ImageOps.expand(pil_img, border=padding_size, fill=(0,))
              # Konversi citra PIL ke array NumPy
              numpy_letter = np.array(padded_letter)
              resize_letter = resize_img(numpy_letter,28,28)
              normalized_letter = resize_letter.reshape(28, 28, 1)

              pred = model.predict(normalized_letter)
              pred_index = np.argmax(pred)
              pred_label = char_list[pred_index]

              letters.append(str(pred_label))
              

              segmented_letters.append(letters)



    total_letter += lps
    roi_index += 1

  return segmented_letters