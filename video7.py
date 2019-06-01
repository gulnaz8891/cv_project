import cv2
import os
import numpy as np


# Path for face image database
path = 'dataset'
recognizer = cv2.face.LBPHFaceRecognizer()


def getImagesWithID(path):
     imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
     faceSamples=[]
     Ids=[]
     for imagePath in imagePaths:
         faceImg=Image.open(imagePath).convert('L')
         faceNp=np.array(faceImg,'unit8')
         ID=int(os.path.split(imagePath)[-1].split('.')[1])
         faces.append(faceNP)
         IDs.append(ID)
         cv2.imshow("training",faceNp)
         cv2.waitKey(10)
     return np.array(IDs), faces

Ids,faces=getImagesWithID(path)
recognizer.train(faces,Ids)
recognizer.save('recognizer/trainningData.yml')
cv2.destroyAllWindows()