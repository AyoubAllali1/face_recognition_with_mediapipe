import cv2
import mediapipe as mp

mpFaceMesh = mp.solutions.face_mesh
mpDraw = mp.solutions.drawing_utils
num_faces = 1
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=num_faces)

img = cv2.imread("image.png")

imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
results = faceMesh.process(imgRGB)
if results.multi_face_landmarks:
    for faceLms in results.multi_face_landmarks:
        mpDraw.draw_landmarks(img, faceLms , mpFaceMesh.FACEMESH_TESSELATION,
                              mpDraw.DrawingSpec(thickness=1,circle_radius=1),
                              mpDraw.DrawingSpec(thickness=1,circle_radius=1))

cv2.imshow('face recognition By Ayoub Allali',img)
cv2.waitKey()

