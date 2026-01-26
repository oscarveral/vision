import cv2 as cv
import numpy as np
import argparse

from rara import dectect_board, PATTERN_SIZE

SQUARE_SIZE = 5.0  # Tamaño del cuadrado en cm

def main():
    parser = argparse.ArgumentParser(
        description="Calibrar la cámara usando un tablero de ajedrez."
    )
    parser.add_argument(
        "--camera_index", type=int, default=0, help="Índice de la cámara (por defecto: 0)"
    )
    parser.add_argument("--width", type=int, default=1280, help="Ancho de la resolución (por defecto: 1280)")
    parser.add_argument("--height", type=int, default=720, help="Alto de la resolución (por defecto: 720)")
    args = parser.parse_args()
    width, height = args.width, args.height

    cap = cv.VideoCapture(args.camera_index)
    cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara.")
        return
    
    real_w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    real_h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    print(f"Resolución solicitada: {width}x{height}")
    print(f"Resolución de la cámara recibida: {real_w}x{real_h}")

    
    cv.namedWindow("Calibration", cv.WINDOW_NORMAL)
    cv.resizeWindow("Calibration", width, height)
    print(f"Iniciando calibración de la cámara para resolución {real_w}x{real_h}.")
    # Preparar puntos 3D del tablero de ajedrez
    tempp = np.zeros((PATTERN_SIZE[0] * PATTERN_SIZE[1], 3), np.float32)
    tempp[:, :2] = np.mgrid[0:PATTERN_SIZE[0], 0:PATTERN_SIZE[1]].T.reshape(-1, 2)
    tempp *= SQUARE_SIZE

    objpoints = []  # Puntos 3D en el espacio del mundo real
    imgpoints = []  # Puntos 2D en la imagen  

    print("---Instrucciones de calibración---")
    print("Muestre el tablero de ajedrez a la cámara desde diferentes ángulos.")
    print("Presione 's' para capturar una imagen cuando el tablero sea detectado.")
    print("Presione 'c' para calibrar la cámara con las imágenes capturadas.")
    print("Presione 'q' para salir sin guardar.")

    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: No se pudo leer el frame de la cámara.")
            break

        found, corners = dectect_board(frame, pattern_size=PATTERN_SIZE, refine_corners=True)
        display_frame = frame.copy()

        if found:
            cv.drawChessboardCorners(display_frame, PATTERN_SIZE, corners, found)
            cv.putText(display_frame, "Tablero detectado. Presione 's' para capturar.", 
                       (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv.putText(display_frame, f"Imágenes capturadas: {count}", 
                   (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv.imshow("Calibration", display_frame)
        key = cv.waitKey(1) & 0xFF

        if key == ord("s") and found:
            objpoints.append(tempp)
            imgpoints.append(corners)
            count += 1
            print(f"Imagen capturada {count}")

        elif key == ord("c"):
            if count < 10:
                print("Se necesitan al menos 10 imágenes capturadas para calibrar.")
                continue

            ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
                objpoints, imgpoints, frame.shape[1::-1], None, None
            )
            if ret:
                print("Calibración exitosa.")
                print(f"Error de reproyección: {ret:.4f}")
                print("Matriz de cámara:\n", mtx)
                print("Coeficientes de distorsión:\n", dist)
                np.savez(f"camera_calibration_{width}x{height}.npz", mtx=mtx, dist=dist)
                print(f"Parámetros de calibración guardados en 'camera_calibration_{width}x{height}.npz'.")
            else:
                print("Error en la calibración.")
        
        elif key == ord("q"):
            print("Saliendo sin guardar.")
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
