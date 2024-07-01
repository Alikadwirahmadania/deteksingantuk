# deteksingantuk
Deteksi ngantuk untuk operator divisi produksi 
import cv2
import dlib
from scipy.spatial import distance
from IPython.display import display, clear_output
from PIL import Image
import io

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

predictor_path = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

(lStart, lEnd) = (42, 48)
(rStart, rEnd) = (36, 42)

cap = cv2.VideoCapture(0)

# Threshold untuk EAR dan jumlah frame berturut-turut
EAR_THRESHOLD = 0.3  # Adjusted threshold
CONSEC_FRAMES = 15  # Reduced number of consecutive frames
frame_counter = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)

        for rect in rects:
            shape = predictor(gray, rect)
            shape = [(shape.part(i).x, shape.part(i).y) for i in range(68)]

            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0

            if ear < EAR_THRESHOLD:
                frame_counter += 1
                if frame_counter >= CONSEC_FRAMES:
                    cv2.putText(frame, "MENGANTUK TERDETEKSI!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(frame, "Perintah untuk memutar suara.", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,  255, 0), 2)
            else:
                frame_counter = 0  # Reset counter jika EAR di atas threshold

        # Tampilkan frame di Jupyter Notebook
        _, img = cv2.imencode('.jpg', frame)
        display(Image.open(io.BytesIO(img.tobytes())))
        clear_output(wait=True)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    pass

cap.release()
cv2.destroyAllWindows()
