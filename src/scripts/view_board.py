import cv2 as cv
from rara import dectect_board, PATTERN_SIZE
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Detectar un tablero de ajedrez en el video de la cámara."
    )
    parser.add_argument(
        "--camera_index", type=int, default=0, help="Índice de la cámara (por defecto: 0)"
    )
    parser.add_argument(
        "--resolution", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"), help="Resolución de la cámara (ancho alto)"
    )
    args = parser.parse_args()
    if args.resolution:
        width, height = args.resolution
    else:
        width, height = 1920, 1080  # Resolución por defecto
    cap = cv.VideoCapture(args.camera_index)
    # cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
    # cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)
    cv.namedWindow("Chessboard", cv.WINDOW_NORMAL)
    cv.resizeWindow("Chessboard", width // 2, height // 2)
    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara.")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: No se pudo leer el frame de la cámara.")
            break

        found, corners = dectect_board(frame, pattern_size=(7, 7), refine_corners=False)
        output_frame = frame.copy()
        if found:
            output_frame = cv.drawChessboardCorners(output_frame, PATTERN_SIZE, corners, found)

        cv.imshow("Chessboard", output_frame)
        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv.destroyAllWindows()