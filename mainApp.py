import cv2
from darkflow.net.build import TFNet
import numpy as np
import time
import knn
import services

option = {
    'model': 'cfg/tiny-yolo-voc-1c-r.cfg',
    'load': -1,
    'threshold': 0.9,
    'gpu': 1.0
}


tfnet = TFNet(option)

capture = cv2.VideoCapture(0)
colors = [tuple(255 * np.random.rand(3)) for i in range(5)]

while (capture.isOpened()):
    stime = time.time()
    ret, frame = capture.read()
    frame = cv2.resize(frame, (1920, 1080))
   
    if ret:
        results = tfnet.return_predict(frame)
        for color, result in zip(colors, results):
            tl = (result['topleft']['x'], result['topleft']['y'])
            br = (result['bottomright']['x'], result['bottomright']['y'])
            label = result['label']
            
            print('Confidence: ' + str(result['confidence']))
            frame = cv2.rectangle(frame, tl, br, color, 7)
            frame = cv2.putText(frame, label, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
            
            if result:
                try:
                    crop_img = frame[tl[1]:br[1], tl[0]-25:br[0]+25]
                    originalImageCrop = crop_img.copy()
                    crop_img = cv2.resize(crop_img, None, fx=2, fy=2)
                    crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
                    crop_img = cv2.GaussianBlur(crop_img, (5,5), 0)
                   
                    crop_img = cv2.bilateralFilter(crop_img, 15, 17, 17)
                    kernel =  cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
                    crop_img = cv2.morphologyEx(crop_img, cv2.MORPH_OPEN, kernel)
                    
                    crop_img = cv2.adaptiveThreshold(crop_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 4)


                    crop_img = cv2.bitwise_not(crop_img)

                    
                    contours, hierarchy = cv2.findContours(crop_img, cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
                    largest_contours = sorted(contours, key=cv2.contourArea)

                    if cv2.contourArea(largest_contours[-2]) < cv2.contourArea(largest_contours[-1]) / 4:
                        largest_contours = largest_contours[-1]
                    else:
                        largest_contours = largest_contours[-2]

                    x,y,w,h = cv2.boundingRect(largest_contours)
                    newCrop = crop_img[y:y+h, x:x+w]
                    

                    print('Plate Number: ')
                    plateNumber = knn.knn(crop_img)
                    print(plateNumber)

                    if plateNumber != 0:
                        if services.checkStolen(plateNumber):
                            print(f"Vehicle number {plateNumber} is stolen!")
                        else:
                            print(f"Vehicle number {plateNumber} is not stolen!")
					

                    cv2.imshow('plate', crop_img)
                    cv2.imshow('newCrop', newCrop)

                except:
                    pass



        cv2.imshow('frame', frame)
        
        print('FPS {:.1f}'.format(1 / (time.time() - stime)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        capture.release()
        cv2.destroyAllWindows()
        break