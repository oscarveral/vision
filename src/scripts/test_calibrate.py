import cv2 as cv
import numpy as np
import time
from pathlib import Path

# Importamos tu detector optimizado
from rara import dectect_board, PATTERN_SIZE, ChessPiece

# --- CONFIGURACIÓN ---
CALIB_FILE = "camera_calibration_1280x720.npz" 
SQUARE_SIZE = 5.0  # cm (Debe coincidir con la calibración)
RESOURCES_PATH = Path("resources")

# Definimos colores (BGR)
COLOR_BASE = (0, 255, 255)   # Amarillo
COLOR_TOP = (0, 165, 255)    # Naranja
COLOR_EDGE = (0, 0, 0)       # Negro (bordes)

def main():
    # [cite_start]1. Cargar Calibración (BT5c) [cite: 232]
    try:
        data = np.load(CALIB_FILE)
        mtx = data['mtx']
        dist = data['dist']
        print(f"Cargada calibración: {CALIB_FILE}")
    except FileNotFoundError:
        print("ERROR: No se encuentra el archivo .npz")
        return

    # 2. Configurar Cámara (IMPORTANTE: MJPEG para mantener 1280x720)
    cap = cv.VideoCapture(2) # Usamos índice 2 como en tu comando anterior
    cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

    # Verificar si la cámara aceptó la resolución
    w_real = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    h_real = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    print(f"Cámara iniciada a {w_real}x{h_real}")

    # Definir el sistema de coordenadas del mundo (Tablero 7x7 esquinas)
    objp = np.zeros((PATTERN_SIZE[0] * PATTERN_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:PATTERN_SIZE[0], 0:PATTERN_SIZE[1]].T.reshape(-1, 2)
    objp = objp * SQUARE_SIZE

    print("AR Iniciado. Pulsa 'q' para salir.")

    # Crear una pieza de ajedrez (peon blanco)
    pawn = ChessPiece("P", RESOURCES_PATH / "P.chsp", color="white")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # [cite_start]3. Detectar Tablero (BT5a) [cite: 230]
        # Usamos tu función optimizada que devuelve True/False rápido
        found, corners = dectect_board(frame)

        if found:
            # [cite_start]4. Estimar Pose (BT5d) [cite: 233]
            # Obtenemos rotación (rvec) y traslación (tvec) de la cámara
            success, rvec, tvec = cv.solvePnP(objp, corners, mtx, dist)
            
            if success:
                # [cite_start]5. Renderizar AR (BT5e) [cite: 235]
                # Dibujamos ejes para referencia
                cv.drawFrameAxes(frame, mtx, dist, rvec, tvec, SQUARE_SIZE*2)
                
                pawn.draw(frame, rvec, tvec, mtx, dist, board_pos=(-1,-1), square_size=SQUARE_SIZE)
                

        cv.imshow("Ajedrez AR - Proyecto Final", frame)
        
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()