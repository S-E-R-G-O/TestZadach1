import cv2
from VideSettings import Processing 
from TrackingBox import Box

detection = []
tracking = []

ob_det = Processing('3.Camera 2017-05-29 16-23-04_137 [3m3s].avi', '4.Camera 2017-05-29 16-23-04_137 [3m3s].avi')


while True: 
    try: 
        contour, frame, thresh = ob_det.detection()
    except Exception as exc: 
        print(exc)
        break

    detection = Box.det_area_create(contour)

    tracking = Box.trackingCreation(detection, tracking)
    Box.histogram(frame, tracking)
    frame = Box.drawBox(frame, tracking)

    cv2.imshow('frame', frame)
    cv2.imshow('thresh', thresh)

    if cv2.waitKey(3) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()