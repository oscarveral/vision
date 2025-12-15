import cv2
import sys
import time
import math
import os # --- NUEVO: Import necesario para manejar rutas ---
from ultralytics import YOLO

# --- CONFIGURACIÓN VISUAL ---
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE_LABEL = 1.4      # Fuente pequeña para etiquetas
FONT_SCALE_TIME = 0.6       # Fuente para el tiempo
COLOR_BOX = (0, 255, 0)     # Verde (B, G, R)
COLOR_TEXT = (0, 0, 0)      # Negro
THICKNESS = 2
# ----------------------------

def run_inference_and_save(weights_path, video_path):
    # 1. Cargar el modelo
    try:
        print(f"🔄 Cargando modelo: {weights_path}...")
        model = YOLO(weights_path)
    except Exception as e:
        print(f"❌ Error al cargar los pesos: {e}")
        return

    # 2. Abrir el vídeo de entrada
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: No se pudo abrir el vídeo '{video_path}'")
        return

    # --- NUEVO: Configuración para GUARDAR el vídeo ---
    # Obtener propiedades del vídeo original para que el de salida sea igual
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Si no puede detectar los FPS, usar 30 por defecto
    if fps == 0 or math.isnan(fps): fps = 30.0

    # Definir el nombre de salida (ej: video.mp4 -> video_procesado.mp4)
    base_name = os.path.basename(video_path)
    name_only, ext = os.path.splitext(base_name)
    output_path = f"{name_only}_procesado.mp4"

    # Definir el codec y crear el objeto VideoWriter
    # 'mp4v' es un buen codec estándar para MP4
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    print(f"💾 El vídeo resultante se guardará en: {output_path}")
    # --------------------------------------------------

    print("▶️  Iniciando inferencia. Presiona 'q' para detener antes.")

    while True:
        success, frame = cap.read()
        if not success:
            break  # Fin del vídeo

        # 3. Medir tiempo de inferencia
        start_time = time.time()
        # verbose=False evita llenar la consola de logs
        results = model(frame, verbose=False) 
        end_time = time.time()
        inference_time_ms = (end_time - start_time) * 1000 # Convertir a ms

        # 4. Dibujar resultados manualmente
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Coordenadas
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Confianza y Clase
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                class_name = model.names[cls]

                # Label
                label = f'{class_name} {conf:.2f}'

                # Dibujar
                cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_BOX, 2)
                
                # Fondo pequeño para el texto
                (w, h), _ = cv2.getTextSize(label, FONT, FONT_SCALE_LABEL, THICKNESS)
                #cv2.rectangle(frame, (x1, y1 - 40), (x1 + w, y1), COLOR_BOX, -1)
                
                # Texto (fuente pequeña)
                #cv2.putText(frame, label, (x1, y1 - 5), FONT, FONT_SCALE_LABEL, COLOR_TEXT, THICKNESS)

        # 5. Mostrar tiempo de respuesta
        info_text = f"Inferencia: {inference_time_ms:.1f} ms"
        
        # Fondo negro esquina superior izquierda
        cv2.rectangle(frame, (0, 0), (220, 40), (0, 0, 0), -1)
        cv2.putText(frame, info_text, (10, 25), FONT, FONT_SCALE_TIME, (0, 255, 255), 1)

        # --- NUEVO: Escribir el frame procesado en el archivo de salida ---
        out_writer.write(frame)
        # -----------------------------------------------------------------

        # 6. Mostrar frame en vivo (opcional, ralentiza un poco el proceso)
        cv2.imshow('Procesando y Guardando...', frame)

        # Salir con 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("⚠️ Proceso interrumpido por el usuario.")
            break

    # Liberar recursos
    cap.release()
    out_writer.release() # --- IMPORTANTE: Cerrar el escritor ---
    cv2.destroyAllWindows()
    print("✅ Proceso finalizado con éxito.")

if __name__ == "__main__":
    # Verificación de argumentos
    if len(sys.argv) < 3:
        print("❌ Uso incorrecto.")
        print("Uso: python probar_y_guardar.py <ruta_pesos.pt> <ruta_video.mp4>")
    else:
        run_inference_and_save(sys.argv[1], sys.argv[2])