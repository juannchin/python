#Detectar rostro desde imagen maestra

import cv2
import face_recognition

# Cargar la imagen maestra y codificar el rostro
imagen_maestra = face_recognition.load_image_file("maestra.jpg")
rostros_maestra = face_recognition.face_encodings(imagen_maestra)

if len(rostros_maestra) == 0:
    print("❌ No se detectó un rostro en la imagen maestra.")
    exit()

rostro_maestro = rostros_maestra[0]  # Tomar el primer rostro detectado

# Lista de imágenes secundarias
imagenes_secundarias = ["foto1.jpg", "foto2.jpg", "foto3.jpg", "foto4.jpg", "foto5.jpg"]

# Verificar cada imagen secundaria
for imagen_path in imagenes_secundarias:
    imagen = face_recognition.load_image_file(imagen_path)
    rostros_en_imagen = face_recognition.face_encodings(imagen)

    # Verificar si el rostro maestro aparece en la imagen secundaria
    for rostro in rostros_en_imagen:
        resultado = face_recognition.compare_faces([rostro_maestro], rostro)
        
        if resultado[0]:  # Si hay coincidencia
            print(f"✅ Se encontró el rostro en {imagen_path}")
            break
    else:
        print(f"❌ No se encontró el rostro en {imagen_path}")
