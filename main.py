import cv2
import numpy as np


N: int = 0
yuv = []
F = []
Y = []
D = []
CT = []
delta = []
w = []
hsv = []
FCM = []
H = []
S = []
Sp = []
Slevel = []
V = []
FV = []
R = []
G = []
B = []
FS = []
intensityAvg = []
FCSV = []
CC = []

# Citirea videoclipului

video = cv2.VideoCapture("resources/lumanare_close.mp4")

# Segmentare in frame-uri

while True:
    (grabbed, frame) = video.read()
    if not grabbed:
        break

    F.append(frame)
    N = N + 1
    cv2.imshow("video initial", frame)
    cv2.waitKey(5)

# Conversie in spatiul YUV pentru detectia flicker-ului

for frame in F:
    yuv.append(cv2.cvtColor(frame, cv2.COLOR_BGR2YUV))

# Calculul mediei de luminanta in fiecare frame

for frame in yuv:
    temp = frame[:, :, 0].astype(np.int16)
    Y.append(temp)
    delta.append(np.average(temp))
    # cv2.imshow("video in spatiu yuv", frame)
    # cv2.waitKey(1)

alpha = (N-2)/(N-1)

# Calculul contributiei luminantei pixelului la totalul frame-ului

for i in range(0, N):
    temp = Y[i]
    temp[temp < delta[i]] = 0
    w.append(temp)

# Calculul derivatei cumulative in functie de timp a luminantei

CT.append(np.ma.zeros((np.shape(Y[1]))))
D.append(np.ma.zeros((np.shape(Y[1]))))

for i in range(1, N):
    di = np.absolute(np.subtract(Y[i], Y[i-1]))
    D.append(di)

    CT.append(alpha*CT[i-1]+(1-alpha)*np.multiply(w[i], D[i]))

    cv2.normalize(CT[i], CT[i], 0, 255, cv2.NORM_MINMAX)

    ave = np.average(CT[i])
    CT[i][CT[i] < ave] = 0

# cv2.imshow("flicker cumulativ", CT[N-1])
# cv2.waitKey(1)

# Conversie in spatiul HSV pentru detectia culorii si separarea pe canale

for frame in F:
    temp = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    H.append(temp[:, :, 0])
    S.append(temp[:, :, 1])
    V.append(temp[:, :, 2])

    B.append(frame[:, :, 0])
    G.append(frame[:, :, 1])
    R.append(frame[:, :, 2])

# Crearea unei masti de culoare in functie de relatia intre canalele RGB

for i in range(0, N):
    temp = np.ma.zeros((np.shape(S[1]))).astype(np.int8)
    temp[(R[i] > 180) & (R[i] > G[i]) & (G[i] > B[i])] = 1
    temp[(R[i] <= 180) | (R[i] <= G[i]) | (G[i] <= B[i])] = 0
    FCM.append(temp)

    # Crearea unei noi matrici de saturatie in functie de masca

    temp = S[i]
    temp[FCM[i] == 0] = 0
    Sp.append(temp)

    # Calculul nivelului mediu de saturatie pe frame

    Slevel.append(np.average(temp[temp > 0]).astype(np.uint8))

    # Calcul apartenenta la foc in functie de saturatie

    temp = S[i]
    mask = [R[i] > 180]
    mat = [[255]*np.size(S[i], 1)]*np.size(S[i], 0)
    if Slevel[i] <= 127:
        temp[mask == 1] = np.subtract(mat, S[i])
    FS.append(temp)
    # cv2.imshow("calcul saturatie", temp)
    # cv2.waitKey(5)

    # Calcul apartenenta la foc in functie de valoare

    temp = V[i]
    intensityAvg.append(np.average(temp[temp > 0]).astype(np.uint8))
    temp[V[i] < max(128, intensityAvg[i])] = 0
    FV.append(temp)

    # Calcul apartenenta la foc combinata din saturatie si valoare

    aux = np.multiply(FS[i], FV[i])
    aux[aux < np.average(aux[aux > 0])] = 0
    FCSV.append(aux)
    # cv2.imshow("calcul apartenenta la candidati de foc", FCSV[i])
    # cv2.waitKey(5)
    cv2.normalize(FCSV[i], FCSV[i], 0, 1, cv2.NORM_MINMAX)

# Calculul apartenentei cumulative in functie de culoare

CC.append(np.ma.zeros((np.shape(S[1]))).astype(np.int8))

for i in range(1, N):
    CC.append(alpha*CC[i-1]+(1-alpha)*FCSV[i])
    # cv2.imshow("culoare cumulativa", CC[i])
    # cv2.waitKey(5)

aux = CC[N-1]
aux[aux < np.average(aux[aux > 0])] = 0

# Calculul apartenentei unui pixel la foc in functie de luminanta si culoare

Fire = np.multiply(CT[N-1], CC[N-1])
cv2.imshow("fire", Fire)
cv2.waitKey(5)

# Unirea regiunilor candidat prin operatii morfologice

dilated_fire = cv2.dilate(Fire, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)), 1)
cv2.imshow("dilated Fire", dilated_fire)
cv2.waitKey(5)

candidate_fire = cv2.morphologyEx(Fire, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8)))
cv2.imshow("candidate Fire", candidate_fire)
cv2.waitKey(0)

# Numararea pixelilor candidat

fire_pixels = np.count_nonzero(candidate_fire)
if fire_pixels > 5000:
    print("fire detected")

# Stergerea capturii video si a ferestrelor

cv2.destroyAllWindows()
video.release()
