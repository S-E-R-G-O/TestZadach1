import cv2

class Processing:
    #Инициализируем переменную первого кадра и захват видео
    def __init__(self, f_name):
        self.firstFrame = None
        self.stream = cv2.VideoCapture(f_name)
    def __del__(self):
        self.stream.release()

    def detect(self):
        ret, frame = self.stream.read()

        if not ret:
            raise Exception('File cannot be open')

        #Обработка фона и применение метода Гауса
        grVideo = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        grVideo = cv2.GaussianBlur(grVideo, (21, 21), 0)

        if self.firstFrame is None:
            self.firstFrame = grVideo
            return [], frame, grVideo #frame

      #Вычесляем абсолютную разность между двумя изображениями(firstFrame и grVideo)
        difference = cv2.absdiff(self.firstFrame, grVideo)

        _, thresh = cv2.threshold(difference, 60, 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, None, iterations=4)
        frameBlur = cv2.blur(thresh, (5, 5))



        cntr, hirarchy = cv2.findContours(frameBlur, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return cntr, frame, thresh