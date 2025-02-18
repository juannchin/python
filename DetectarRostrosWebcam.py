#Detectar rostros desde Webcam

import cv2
import face_recognition

# Cargar imágenes de referencia y obtener sus codificaciones faciales
image1 = face_recognition.load_image_file("persona-1.jpg")  # Cambia por tus imágenes
image2 = face_recognition.load_image_file("persona-2.jpg")

# Obtener la codificación facial de cada imagen
encoding1 = face_recognition.face_encodings(image1)[0]
encoding2 = face_recognition.face_encodings(image2)[0]

# Lista de rostros conocidos y sus nombres
known_encodings = [encoding1, encoding2]
known_names = ["Juan", "Maria"]

# Iniciar la webcam
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        break
    
    # Convertir el frame a RGB (OpenCV usa BGR por defecto)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detectar rostros en el frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Comparar con los rostros conocidos
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        
        name = "Desconocido"
        if True in matches:
            match_index = matches.index(True)
            name = known_names[match_index]

        # Dibujar un rectángulo alrededor del rostro detectado
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Mostrar el video en tiempo real
    cv2.imshow("Detección de Rostros", frame)

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar ventanas
video_capture.release()
cv2.destroyAllWindows()
