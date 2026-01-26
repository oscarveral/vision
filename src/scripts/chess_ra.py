import time
import cv2
import numpy as np
import argparse
import os
import yaml # pip install PyYAML
from pathlib import Path

from rara import dectect_board, PATTERN_SIZE, Chessboard

# --- FUNCIONES DE INTERFAZ (HUD) ---
def draw_text_with_bg(img, text, pos, font_scale=0.6, text_color=(255, 255, 255), bg_color=(0, 0, 0)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    (t_w, t_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = pos
    
    # Fondo
    sub_img = img[y:y+t_h+10, x:x+t_w+10]
    white_rect = np.full(sub_img.shape, bg_color, dtype=np.uint8)
    res = cv2.addWeighted(sub_img, 0.4, white_rect, 0.6, 1.0)
    img[y:y+t_h+10, x:x+t_w+10] = res
    
    # Texto
    cv2.putText(img, text, (x+5, y+t_h+5), font, font_scale, text_color, thickness, cv2.LINE_AA)

def draw_hud(frame, status_msg, input_mode, input_buffer, input_prompt):
    h, w, _ = frame.shape
    
    # Instrucciones
    instructions = [
        "CONTROLES:",
        "[x] Escribir movimiento (e.g. e2e4)",
        "[u] Deshacer movimiento",
        "[r] Resetear tablero",
        "[l] Cargar FEN desde fichero",
        "[q] Salir"
    ]
    
    start_y = 20
    for line in instructions:
        draw_text_with_bg(frame, line, (10, start_y), font_scale=0.5, bg_color=(50, 50, 50))
        start_y += 25

    # Mensaje de Estado
    if status_msg:
        draw_text_with_bg(frame, f"STATUS: {status_msg}", (10, h - 40), font_scale=0.7, bg_color=(0, 0, 150), text_color=(0, 255, 255))

    # Caja de Entrada
    if input_mode:
        box_w, box_h = 400, 60
        x1 = (w - box_w) // 2
        y1 = (h - box_h) // 2
        
        cv2.rectangle(frame, (x1, y1), (x1 + box_w, y1 + box_h), (0, 255, 0), 2)
        cv2.rectangle(frame, (x1, y1), (x1 + box_w, y1 + box_h), (0, 0, 0), -1)
        
        display_text = f"{input_prompt}: {input_buffer}_"
        cv2.putText(frame, display_text, (x1 + 10, y1 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

def load_config(config_path):
    """Carga el archivo YAML de configuración."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"No se encontró el archivo de configuración: {config_path}")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Chess AR")
    parser.add_argument("--config", type=str, default="config.yaml", help="Ruta al archivo de configuración YAML.")
    args = parser.parse_args()

    # 1. CARGAR CONFIGURACIÓN
    try:
        cfg = load_config(args.config)
        print("Configuración cargada correctamente.")
    except Exception as e:
        print(f"Error cargando configuración: {e}")
        return

    # Extraer variables para facilitar uso
    calib_file = cfg['camera']['calibration_file']
    square_size = 5.0*cfg['game']['scale']
    pieces_paths = cfg['assets']['pieces']
    positions_folder = cfg['positions']['folder']
    initial_pos_file = cfg['positions']['initial_file']

    # 2. CARGAR CALIBRACIÓN DE CÁMARA
    with np.load(calib_file) as data:
        mtx, dist = data['mtx'], data['dist']

    # 3. INICIALIZAR JUEGO
    game = Chessboard(mtx, dist, square_size, pieces_paths, nrots=20)
    
    # Cargar posición inicial desde la carpeta configurada
    start_path = os.path.join(positions_folder, initial_pos_file)
    if initial_pos_file == 'default':
        game.reset_board()
    elif os.path.exists(start_path):
        try:
            game.load_fen_from_file(start_path)
        except Exception as e:
            print(f"Error cargando posición inicial. Usando tablero estándar. Error: {e}")
            game.reset_board()
    else:
        print(f"Advertencia: Archivo inicial {start_path} no encontrado. Usando tablero estándar.")
        game.reset_board()

    # 4. INICIALIZAR CÁMARA
    cap = cv2.VideoCapture(cfg['camera']['index'])
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg['camera']['width'])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg['camera']['height'])

    # Definir mundo 3D
    objp = np.zeros((PATTERN_SIZE[0] * PATTERN_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:PATTERN_SIZE[0], 0:PATTERN_SIZE[1]].T.reshape(-1, 2)
    objp *= square_size

    # Variables de estado
    last_rvec, last_tvec = None, None
    missed_frames = 0
    MAX_MISSED_FRAMES = 8
    prev_corners = None
    
    # Variables UI
    status_message = "Sistema listo. Pulsa 'x' para mover."
    status_timer = 0
    input_mode = False
    input_buffer = ""
    input_target = "MOVE"

    while True:
        tic = time.time()
        ret, frame = cap.read()
        if not ret: break

        found, corners = dectect_board(frame, prev_corners=prev_corners)
        
        if found:
            success, rvec, tvec = cv2.solvePnP(objp, corners, mtx, dist)
            prev_corners = corners
            
            if success:
                if last_rvec is not None:
                    rvec = 0.6 * last_rvec + 0.4 * rvec
                    tvec = 0.6 * last_tvec + 0.4 * tvec
                
                game.draw(frame, rvec, tvec)
                last_rvec, last_tvec = rvec, tvec
                missed_frames = 0
        else:
            missed_frames += 1
            if missed_frames < MAX_MISSED_FRAMES and last_rvec is not None:
                game.draw(frame, last_rvec, last_tvec)
            else:
                prev_corners = None

        # --- UI ---
        if status_timer > 0:
            status_timer -= 1
        else:
            status_message = ""

        prompt = "Movimiento (UCI)" if input_target == "MOVE" else "Posicion (FEN archivo)"
        draw_hud(frame, status_message, input_mode, input_buffer, prompt)
        cv2.imshow("Chess AR Configurable", frame)
        
        # --- INPUT ---
        key = cv2.waitKey(1) & 0xFF

        if input_mode:
            if key == 13: # Enter
                input_mode = False
                if input_target == "MOVE":
                    if game.make_move(input_buffer):
                        status_message = f"Movimiento {input_buffer} OK"
                    else:
                        status_message = "Movimiento ILEGAL"
                elif input_target == "FILE":
                    # Construir ruta completa usando la carpeta configurada
                    full_path = os.path.join(positions_folder, input_buffer+'.fen')
                    try:
                        game.load_fen_from_file(full_path)
                    except Exception as e:
                        status_message = f"Error cargando posición: {e}"
                
                status_timer = 150
                input_buffer = ""
            elif key == 8 or key == 127: # Backspace
                input_buffer = input_buffer[:-1]
            elif key == 27: # Esc
                input_mode = False
                input_buffer = ""
                status_message = "Cancelado"
            elif key < 255 and key != 0:
                try:
                    char = chr(key)
                    if char.isalnum() or char in "./_-": 
                        input_buffer += char
                except: pass
        else:
            if key == ord('q'): break
            elif key == ord('x'):
                input_mode = True
                input_target = "MOVE"
                input_buffer = ""
            elif key == ord('l'):
                input_mode = True
                input_target = "FILE"
                input_buffer = ""
            elif key == ord('u'):
                if game.undo_move(): status_message = "Deshacer OK"
                else: status_message = "Nada que deshacer"
                status_timer = 100
            elif key == ord('r'):
                game.reset_board()
                status_message = "Reset OK"
                status_timer = 100

        toc = time.time()
        #print(f"Frame Time: {(toc - tic)*1000:.2f} ms")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()