import cv2  # Импортируем библиотеку OpenCV для работы с изображениями и видео

class Processing:
    def __init__(self,  fileName1, fileName2):
        # Инициализируем объект класса
        self.firstName = None  # Хранит первый кадр в градациях серого
        self.stream1 = cv2.VideoCapture(fileName1)  # Открываем видеофайл с указанным именем
        #self.firstName = None  # Хранит первый кадр в градациях серого
        self.stream2 = cv2.VideoCapture(fileName2)
       
        
    def __del__(self):
        # Освобождаем ресурсы при уничтожении объекта
        self.stream1.release()  # Закрываем захват видео, чтобы освободить ресурсы
        self.stream2.release()

    def detection(self):
        # Метод для обнаружения изменений между кадрами
        ret1, frame1 = self.stream1.read() 
        ret2, frame2 = self.stream2.read() # Читаем следующий кадр из видео
        #Объеденить два окна в один
        frame = cv2.hconcat([frame1, frame2])


        if not ret1 or not ret2:
            # Проверка на успешность чтения кадра
            raise Exception("Failed to open video file")  # Если не удалось получить кадр, выбрасываем исключение
            
        # Преобразуем цветное изображение в градации серого
        grVideo = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Применяем гауссово размытие для уменьшения шума
        grVideo = cv2.GaussianBlur(grVideo, (21, 21), 0)

        if self.firstName is None:
            # Если первый кадр еще не установлен, сохраняем текущий
            self.firstName = grVideo
            return [], frame, grVideo  # Возвращаем пустой список контуров, оригинальный кадр и серый кадр

        # Вычисляем абсолютную разницу между первым кадром и текущим
        difference = cv2.absdiff(self.firstName, grVideo) 
        # Применяем пороговое значение для выделения значительных изменений
        _, thresh = cv2.threshold(difference, 60, 255, cv2.THRESH_BINARY)
        # Дилатация для увеличения областей (очистка шума)
        thresh = cv2.dilate(thresh, None, iterations=4)
        # Дополнительное размытие для упрощения находящихся контуров
        frameBlur = cv2.GaussianBlur(thresh, (5, 5), 0) 

        # Находим контуры изменений
        cntr, hiracachy = cv2.findContours(frameBlur, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return cntr, frame, thresh  # Возвращаем найденные контуры, оригинальный кадр и бинарное изображение
