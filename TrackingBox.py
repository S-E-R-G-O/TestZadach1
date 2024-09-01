from statistics import mean

import cv2
import numpy as np
import HungarianAlgorithm as hm


class Box:
    Green_Clr = (100, 255, 0)  # Зеленый
    Red_Clr = (0, 0, 255)  # Красный
    det_limArrea = 8000
    track_hist = {}
    del_hists = {}
    id_counter = 0

    def __init__(self, x, y, w, h):
        self.x, self.y, self.w, self.h = x, y, w, h
        self.id = Box.id_counter
        Box.id_counter += 1

    def shape(self):
        return self.x, self.y, self.w, self.h

    def rectangle(self):
        return [self.x, self.y, self.w + self.x, self.h + self.y]

    @classmethod
    def det_area_create(cls, contours):
        return [cls(*cv2.boundingRect(cnt)) for cnt in contours if cv2.contourArea(cnt) > cls.det_limArrea]

    @classmethod
    def trackingCreation(cls, detections, trackers):
        if not trackers:
            return detections
        if not detections:
            for t in trackers:
                if t.id in cls.track_hist:
                    cls.del_hists.update({t.id: cls.track_hist.pop(t.id)})
                else:
                    cls.del_hists.update({t.id: []})
                print("DEL", t.id, len(cls.del_hists[t.id]))
                del t
            return []
            # Формируем матрицу весов IntersectionOverUnion
        IoU = np.array([[hm.IntersectionOverUnion(det, trk) for trk in trackers] for det in detections],
                       dtype=np.float32)
        # Пропускаем матрицу весов IntersectionOverUnion через венгерский алгоритм
        matches, unmatched_det, unmatched_trk = hm.hungarian(IoU, trackers, detections)
        # matches - Список для пар найденных сопоставлений
        # unmatched_det - Список для несопоставленных детекций
        # unmatched_trk - Список для несопоставленных трекеров, отслеживаемый объект пропал из кадра - надо удалить

        # через матрицу соответстви делаем отдетектированные объекты, тречными
        for mtcd in matches:
            # id отдетектированного объекта, делаем тем же, чо и отслеживаемого
            detections[mtcd[0]].id = trackers[mtcd[1]].id
            # детектируемый объект становится обновленным отслежеваемым
            trackers[mtcd[1]] = detections[mtcd[0]]

        # добавляем нераспределенне отдетектированные объект к новым отслеживаемые
        trackers.extend(detections[i] for i in unmatched_det)


        # удаляем объекты, которые мы тречили, но их нет в новом кадре, значит они исчезли
        for ut in sorted(unmatched_trk, reverse=True):
            # Записываем массив гистограм удаляемого оюъекта, ключ - id удаленного
            if trackers[ut].id in cls.track_hist:
                cls.del_hists.update({trackers[ut].id: cls.track_hist.pop(trackers[ut].id)})
            else:
                cls.del_hists.update({trackers[ut].id: []})
            # Удаляем трекер
            print("DEL", trackers[ut].id, len(cls.del_hists[trackers[ut].id]))
            del trackers[ut]

        return trackers

    @classmethod
    def histogram(cls, frame, boxes):
        frame_height, frame_width = frame.shape[0], frame.shape[1]
        # Проходим по каждому ограничивающему прямоугольнику (box) в переданных рамках
        for box in boxes:
            # Извлекаем координаты и размеры текущего ограничивающего прямоугольника
            x, y, w, h = box.shape()  # Предполагается, что box.shape() возвращает (x, y, width, height)
            if y == 0 or y + h >= frame_height:
                return
            # Обрезаем изображение (кадр) по координатам ограничивающего прямоугольника
            # Верх и низ объекта 
            tr_frame_top = frame[y: y + h, x: x + w]
            tr_frame_bot = frame[y + h: frame_height, x: x + w]

            # Вычисляем цветовую гистограмму для обрезанного кадра
            new_hist_top = cv2.calcHist([tr_frame_top],
                                    [0, 1, 2],  # Каналы цветов (BGR)
                                    None,  # Не используем маску
                                    [8, 8, 8],  # Число бинов для каждого канала
                                    [0, 256, 0, 256, 0, 256])  # Диапазоны значений цвета

            # Нормализуем гистограмму, чтобы сумма всех значений была равна 1
            cv2.normalize(new_hist_top, new_hist_top)

            new_hist_bot = cv2.calcHist([tr_frame_bot],[0, 1, 2],  # Каналы цветов (BGR)
                                    None,  # Не используем маску
                                    [8, 8, 8],  # Число бинов для каждого канала
                                    [0, 256, 0, 256, 0, 256])
            cv2.normalize(new_hist_bot, new_hist_bot)

            # Сохраняем новую гистограмму

            if box.id in cls.track_hist:
                cls.track_hist[box.id].append(new_hist_top)
                cls.track_hist[box.id].append(new_hist_bot)
            else:
               
                print("NEW", box.id)
                hist_weight_top = cls.compare_histograms(new_hist_top)
                hist_weight_bot = cls.compare_histograms(new_hist_bot)

                if hist_weight_top:
                    for hw_id in hist_weight_top:
                        # среднее значение корреляции между новой и старой гистограммой
                        print(f"ВЕРХНЯЯ ГИСТОГРАММА: {hw_id}: {mean(hist_weight_top[hw_id])}")
                cls.track_hist[box.id] = [new_hist_top]

                if hist_weight_bot:
                    for hw_id in hist_weight_bot:
                        # среднее значение корреляции между новой и старой гистограммой
                        print(f"НИЖНЯЯ ГИСТОГРАММА: {hw_id}: {mean(hist_weight_bot[hw_id])}")
                cls.track_hist[box.id].append(new_hist_bot)
    @classmethod
    def compare_histograms(cls, new_hist):
           # Создаем словарь для хранения значений корреляции между новым гистограммой и старыми
        hist_weight = {}
         # Если нет ни одной сохраненной гистограммы, возвращаем None
        if len(cls.del_hists) == 0:
            return None
        for i in cls.del_hists:
             # Сравниваем new_hist с каждой сохранённой гистограммой с помощью метода cv2.compareHist
            for hist in cls.del_hists[i]:
                res = cv2.compareHist(new_hist, hist, cv2.HISTCMP_CORREL)
                 # Если ключ i уже присутствует в словаре hist_weight, добавляем результат в соответствующий список
                if i in hist_weight:
                    hist_weight[i].append(res)
                else:
                     # Если ключа i нет, создаем новый список с первым результатом
                    hist_weight[i] = [res]
            
    # Возвращаем словарь hist_weight, содержащий степени корреляции для каждой ключевой гистограммы
        return hist_weight

    @classmethod
    def drawBox(cls, frame, boxes):
        for box in boxes:
            x, y, w, h = box.shape()
            cx, cy = x + w // 2, y + h // 2
            cv2.putText(frame, str(box.id), (cx, cy - 7), 0, 0.5, cls.Red_Clr, 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), cls.Green_Clr, 2)
            #cv2.circle(frame, (cx, cy), 2, cls.Red_Clr, -1)
            #Отрисовка прямой линии в центре коробки от одной границы до другой 
            cv2.line(frame, (x, cy), (x + w, cy), cls.Red_Clr, 2)
        return frame
