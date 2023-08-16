import cv2,glob
 
all_images=glob.glob("*.jpg")
detect   = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

for image in all_images:
        img = cv2.imread(image)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detect.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)
 
        for (x,y,w,h) in faces:
                # To draw a rectangle in a face
                final_img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
 
        cv2.imshow("Face Detection",final_img)
        k = cv2.waitKey(2000)
        
 
        # De-allocate any associated memory usage
        cv2.destroyAllWindows()
