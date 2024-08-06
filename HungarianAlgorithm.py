import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment

#Согласно венгерскому методу: задаем строки(detection) и столбцов(trackers) и метрику пересечения
#между рамками iou_third = 0.3 - коэфицент границы
def hungarian(IoU, trackers, detections, iou_third = 0.3):
    detections_idx, trackers_idx = linear_assignment(-IoU)

    #print("detections_idx", detections_idx)
    #print("trackers_idx  ", trackers_idx)

    if len(detections_idx) == 0:
        return [], [], []

    #Объявляем массивы для нераспределенных столбцов и строк они возникают в том случае если
    #не был превышен iou_third
    unmatched_trackers, unmatched_detections = [], []

    #Заполняем списки нераспределенных столбцов(unmatched_trackers) и
    # строк(unmatched_detections)
    for t, trk in enumerate(trackers):
        if t not in trackers_idx:
            unmatched_trackers.append(t)
    for d, det in enumerate(detections):
        if d not in detections_idx:
            unmatched_detections.append(d)

    matches = []
    # Ecли IoU меньше порогового значения то считаем, что объект другой и присваеваем ему новый id
    for i, _ in enumerate(detections_idx):
        if IoU[detections_idx[i], trackers_idx[i]] < iou_third:
            unmatched_trackers.append(trackers_idx[i])
            unmatched_detections.append(detections_idx[i])
        else:
            matches.append([detections_idx[i], trackers_idx[i]])

    if len(matches) == 0:
        matches = np.empty((0,2), dtype=int)
    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

 #Определение (x, y)-координаты прямоугольника пересечения
def IntersectionOverUnion(a,b):
    boxA = a.rectangle()
    boxB = b.rectangle()
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    #Формула вычисления площади прямоугольника пересечения
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # вычисление площади предсказанного и истинного прямоугольника
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    #Вычесляем iou: Область пересечения делим на (сумму прогнозируемого значения и истинного
    #вычитая при этом область пересечения)
    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou




