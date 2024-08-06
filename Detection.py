import cv2
import numpy as np
from TrackingBox import Box
from VideSettings import Processing

detection = []  # Детектированные объекты
tracking =[] # Отслеживаем объекты
ob_det = Processing("3.Camera 2017-05-29 16-23-04_137 [3m3s].avi") # Передаем видео и создаем его маску из VideSettings

while True:
    try:
        contour, frame, thresh = ob_det.detect()
    except Exception as exc:
        print(exc)
        break
    detection = Box.det_area_create(contour)
    tracking = Box.trackingCreation(detection, tracking)
    Box.histogram(frame,tracking)

    frame = Box.drawing_box(frame,tracking)

    cv2.imshow("Tracking", frame)
    cv2.imshow("Mask", thresh)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
print('histo', Box.del_hists)
cv2.destroyAllWindows()