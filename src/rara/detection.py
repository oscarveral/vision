import cv2 as cv
import numpy as np

PATTERN_SIZE = (7, 7)

def dectect_board(image, pattern_size=PATTERN_SIZE, refine_corners=True, prev_corners=None):
    """
    Detecta el tablero y devuelve las coordenadas refinadas de las esquinas.

    Args:
        image: Imagen en la que se detectará el tablero. BGR o GRAY.
        pattern_size: Tupla que indica el número de cuadros internos en el tablero de ajedrez.
        refine_corners: Booleano para decidir si refinar las esquinas detectadas.
        prev_corners: Esquinas detectadas en el frame anterior. Usado para la refinación de la orientación.

    Returns:
        found: Booleano que indica si se encontró el tablero.
        corners: Coordenadas de las esquinas detectadas (si se encuentra el tablero). 
    """

    if len(image.shape) == 3:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        gray = image

    # Configuración de flags para findChessboardCorners
    # - ADAPTIVE_THRESH: Ayuda si la iluminación no es uniforme.
    # - FAST_CHECK: Si no ve un tablero, pasa rápido al siguiente frame (evita lag).
    # - NORMALIZE_IMAGE: Mejora el contraste antes de detectar.
    flags = cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_FAST_CHECK

    # Redimensionamos para acelerar la detección en imágenes grandes
    scale_factor = 0.3
    small_gray = cv.resize(gray, (0, 0), fx=scale_factor, fy=scale_factor)

    found_small, _ = cv.findChessboardCorners(small_gray, pattern_size, flags)
    if not found_small:
        return False, None

    # Si se detecta en la imagen pequeña, buscar en la original
    found, corners = cv.findChessboardCorners(gray, pattern_size, flags)
    if not found:
        return False, None
    
    if refine_corners:
        # Refinar la posición de las esquinas para mayor precisión (BT5i).
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        #corners = sort_corners(corners)#, prev_corners)
        corners = sort_corners_hist(corners, pattern_size=pattern_size, prev_corners=prev_corners)
    return True, corners

def sort_corners(corners, pattern_size=(7, 7)):
    """
    Reordena las esquinas para garantizar que el sistema de coordenadas 
    esté alineado con la pantalla:
    - Origen (0,0): Arriba-Izquierda visual.
    - Eje X (filas): Apunta a la Derecha visual.
    - Eje Y (cols): Apunta Abajo visual.
    
    Esto corrige rotaciones de 90, 180 y 270 grados automáticamente.
    """
    # 1. Asegurar que el punto 0 es el Top-Left visual (mínima suma X+Y)
    # Esto ya arregla grandes inversiones, pero no la dirección de los ejes.
    pts = corners.reshape(-1, 2)
    sums = pts.sum(axis=1)
    tl_idx = np.argmin(sums)
    
    # Si el punto 0 no es el TL visual, rotamos los datos hasta que lo sea.
    # Pero como findChessboardCorners devuelve un grid estructurado, 
    # es más robusto analizar la estructura local.
    
    # Trabajamos con el grid como matriz 2D (7x7) para poder transponer/rotar
    grid = corners.reshape(pattern_size[1], pattern_size[0], 2)
    
    # --- PASO 1: CORREGIR ROTACIÓN DE 90 GRADOS (Transposición) ---
    # Analizamos el vector del primer segmento (0,0) -> (0,1) [Primera fila, segundo punto]
    v_row = grid[0, 1] - grid[0, 0]
    
    # Si el cambio en Y es mayor que en X, es que las "filas" van hacia abajo.
    # Eso es un eje X vertical. ¡Mal! -> Transponemos para que sean filas reales.
    if abs(v_row[1]) > abs(v_row[0]):
        # Transponer: Cambiamos filas por columnas (swapaxes)
        grid = grid.transpose(1, 0, 2)
        # Recalculamos vector tras transponer
        v_row = grid[0, 1] - grid[0, 0]

    # --- PASO 2: CORREGIR DIRECCIÓN HORIZONTAL (Eje X a la Derecha) ---
    # Si v_row[0] es negativo, el eje X va a la izquierda.
    if v_row[0] < 0:
        # Invertimos el orden de las columnas (Espejo horizontal)
        grid = np.flip(grid, axis=1)

    # --- PASO 3: CORREGIR DIRECCIÓN VERTICAL (Eje Y Abajo) ---
    # Analizamos vector de columna (0,0) -> (1,0) [Segunda fila, primer punto]
    v_col = grid[1, 0] - grid[0, 0]
    
    # Si v_col[1] es negativo, el eje Y va hacia arriba.
    if v_col[1] < 0:
        # Invertimos el orden de las filas (Espejo vertical)
        grid = np.flip(grid, axis=0)

    # Devolvemos aplanado al formato que espera solvePnP
    return grid.reshape(-1, 1, 2)

def sort_corners_hist(corners, pattern_size=(7, 7), prev_corners=None):
    """
    Ordena las esquinas probando las 8 posibles permutaciones del grid
    para encontrar la que minimiza la distancia al frame anterior.
    """

    current_grid = corners.reshape(pattern_size[1], pattern_size[0], 2)

    if prev_corners is None:
        # Sin frame previo, usar la versión básica
        return sort_corners(corners, pattern_size=pattern_size)
    
    else:
        prev_grid = prev_corners.reshape(pattern_size[1], pattern_size[0], 2)

        ref_points = np.array([
            prev_grid[0,0],  # Top-Left
            prev_grid[0,-1], # Top-Right
            prev_grid[-1,0], # Bottom-Left
            prev_grid[-1,-1] # Bottom-Right
        ])

        best_grid = None
        min_distance = float('inf')

        bases = [current_grid, current_grid.transpose(1,0,2)]

        for base in bases:
            for k in range(4):
                candidate = np.rot90(base, k=k)
                cand_points = np.array([
                    candidate[0,0],
                    candidate[0,-1],
                    candidate[-1,0],
                    candidate[-1,-1]
                ])
                dist = np.linalg.norm(cand_points - ref_points)
                if dist < min_distance:
                    min_distance = dist
                    best_grid = candidate

        return best_grid.reshape(-1, 1, 2)