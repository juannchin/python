import cv2
from deepface import DeepFace

#Detectar emociones desde Webcam

# Inicializar la cámara
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Mostrar el video en tiempo real
    cv2.imshow("Webcam - Presiona 'q' para salir", frame)

    try:
        # Analizar la emoción de la cara detectada
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        
        # Extraer la emoción dominante
        if isinstance(result, list):
            emotion = result[0]['dominant_emotion']
        else:
            emotion = result['dominant_emotion']

        # Mostrar la emoción en la ventana
        cv2.putText(frame, f"Emocion: {emotion}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    except Exception as e:
        print("Error en la detección:", e)

    # Mostrar el video con la emoción detectada
    cv2.imshow("Detección de Sentimiento", frame)

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar ventanas
cap.release()
cv2.destroyAllWindows()
