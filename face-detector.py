import cv2

human_cascade = cv2.CascadeClassifier('human_face.xml')
cat_cascade = cv2.CascadeClassifier('cat_face.xml')

img = cv2.imread(input("Enter file path (with file extension): "))

scale_percent = 100
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)

img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA) 

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


human_face = human_cascade.detectMultiScale(img_gray, 1.1, 5)
cat_face = cat_cascade.detectMultiScale(img_gray, 1.1, 4)

for (i, (x, y, w, h)) in enumerate(human_face):
    cv2.rectangle(img, (x, y), (x+w, y+h), (191,62,255), 2)
    cv2.putText(img, "Human Face {}.".format(i + 1), (x + 10, y - 20),
                cv2.FONT_HERSHEY_DUPLEX, 0.40, (191,62,255), 1)


for (i, (x, y, w, h)) in enumerate(cat_face):
    cv2.rectangle(img, (x, y), (x+w, y+h), (0,238,238), 2)
    cv2.putText(img, "Cat Face {}.".format(i + 1), (x + 10, y - 20),
                cv2.FONT_HERSHEY_DUPLEX, 0.40, (0,238,238), 1)

cv2.imshow("Face Detector",img)

key = cv2.waitKey()

print("Press s to save image")
print("Press esc to quit")

name = input("Enter a file name (with extension): ")

if key == 115 or key == 83:
    cv2.imwrite('Detected-face/{}'.format(name),img)
    print("Image saved successfuly")
elif key == 27:
    cv2.destroyAllWindows()
