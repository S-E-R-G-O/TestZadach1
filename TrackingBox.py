import cv2
import numpy as np
import HungarianAlgorithm as HA
class Box:
    Green_Clr = (100,255,0) # Палитра зелёного цвета в BGR
    Red_Clr = (0,0,255) # Палитра красного цвета в BGR
    lim_detArea = 8000 # Граничное значение при котором происходит отрисовка объекта
    id = 0 # Номер id объекта обновляется при появлении нового объекта в кадре
    track_hists = {} 
    del_hists = {} 


    #Инициализируем координаты рамки и id которое увеличивается каждый раз
    #когда появляется новый экземпляр класса BOX
    def __init__(self,x,y,w,h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.id = Box.id
        Box.id += 1
    # Возвращение размеров прямоугольника
    def shape(self):
        return self.x, self.y, self.w, self.h

    def rectangle(self):
        return [self.x, self.y, self.w + self.x, self.h + self.y]

    #Задаем условия отрисовки рамки
    @classmethod
    def det_area_create(cls, contour):
        detection = []
        if len(contour) != 0:
            for cnt in contour:
                area = cv2.contourArea(cnt) #Вычесление площади вокруг объекта

            #Объявляем условие отрисовки рамки вокруг объекта
                if area > cls.lim_detArea:
                    x, y, w, h = cv2.boundingRect(cnt)
                    detection.append(cls(x,y,w,h))
        return detection

    @classmethod
    def trackingCreation(cls, detection, tracking):
        if len(tracking) == 0:
            return detection
        if len(detection) == 0:
            return []

        # Формирование матрицы весов IoU - intersection over union
        IoU = np.zeros((len(detection), len(tracking)), dtype=np.float32)



        #Выполнение перебора всех распознаных объектов (detection) и всех отслеживаемых объектов (tracking)
        # и для каждой пары объектов вычисляется площадь пересечения их областей (Intersection over Union).
        for i, det in enumerate(detection):
            for j, tra in enumerate(tracking):
                IoU[i][j] = HA.IntersectionOverUnion(det,tra)
        
        
        
        matches, unmatched_detections, unmatched_trackers = HA.hungarian(IoU,tracking,detection)
        
        for mtc in matches:
            #Сопоставляем идентификатор обнаруженного объекта с отслеживаемым объектом.
            detection[mtc[0]].id = tracking[mtc[1]].id

            tracking[mtc[1]] = detection[mtc[0]]

        #Нераспознанный объект становится новым отслеживаемым
        for i in unmatched_detections:
            tracking.append(detection[i])

       
        u_t = -np.sort(-unmatched_trackers)
        #Обновляем список удаленны гистограмм
        for i in u_t:
            Box.del_hists.update({tracking[i].id:Box.track_hists.pop(tracking[i].id)})
            del tracking[i]
        return tracking
    

    #Создаем функцию формирования гистограмм
    @classmethod
    def histogram(cls, frame, boxes):
        if len(boxes) > 0:
            for box in boxes:
                x, y, w, h = box.shape()
                tr_frame = frame[y:y+h, x:x+w]
                #Строим гистограмму для объекта
                hist = cv2.calcHist([tr_frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
                cv2.normalize(hist, hist).flatten()
                
                #Записываем гистограмму текущего кадра в объект 
                if box.id in cls.track_hists: 
                    cls.track_hists[box.id].append(hist)
                else:
                    cls.track_hists[box.id] = [hist]

     


    @classmethod
    #Отрисовываем рамку и точку-центр внутри рамки
    def drawing_box(cls, frame, boxes):
        if len(boxes) > 0:
            for box in boxes:
                x, y, w, h = box.shape()
                cx = x + w // 2
                cy = y + h // 2
                cv2.putText(frame, str(box.id),(cx, cy -7), 0,0.5, cls.Red_Clr, 2)
                cv2.rectangle(frame,(x,y),(x+w, y+h), cls.Green_Clr,2)
                cv2.circle(frame, (cx,cy), 2, cls.Red_Clr, -1)
        return frame