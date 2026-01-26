import cv2 as cv
import numpy as np
import chess

class ChessPiece:
    def __init__(self, name, profile_path, nrots=12, color='white'):
        """
        Args:
            name: Nombre de la pieza (e.g. 'P' para peón).
            profile_path: Ruta al archivo de perfil (.chsp).
            nrots: Número de rotaciones para el modelo 3D.
            color: 'white' o 'black' para definir el color base de la pieza.
        
        Raises:
            ValueError: Si el color no es 'white' o 'black'.
        """

        if color not in ['white', 'black']:
            raise ValueError("color debe ser 'white' o 'black'")

        self.name = name
        self.color_type = color
        self.nrots = nrots
        self.profile = self._load_profile(profile_path)

        if self.color_type == 'white':
            self.base_color = np.array([220, 240, 255], dtype=np.float32) # Crema / Beige (BGR)
        else:
            self.base_color = np.array([0, 24, 48], dtype=np.float32)    # Gris oscuro (BGR)
        
        # Generar geometría basado en el perfil y el número de rotaciones
        # Almacenamos vértices y caras como numpy arrays para eficiencia
        self.vertices, self.faces = self._generate_geometry()

    def _load_profile(self, profile_path):
        points = []
        try:
            with open(profile_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        points.append((float(parts[0]), float(parts[1])))
        except FileNotFoundError:
            return [(0,0), (1,0), (1,5), (0,5)] 
        return points

    def _generate_geometry(self):
        """
        Genera los vértices y caras del modelo 3D mediante revolución del perfil.
        
        Args:
            None

        Returns:
            vertices: Array numpy de vértices 3D.
            faces: Array numpy de índices de caras (triángulos).
        """
        vertices = []
        faces = [] 
        n_profile = len(self.profile)
        angles = np.linspace(0, 2 * np.pi, self.nrots, endpoint=False)
        
        # Calcular vértices aplicando revolución
        for r, z in self.profile:
            for theta in angles:
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                vertices.append([x, y, -z]) 
        
        vertices = np.array(vertices, dtype=np.float32)

        # Generar caras
        for i in range(n_profile - 1): 
            for j in range(self.nrots): 
                current = i * self.nrots + j
                next_rot = i * self.nrots + (j + 1) % self.nrots
                below = (i + 1) * self.nrots + j
                below_next = (i + 1) * self.nrots + (j + 1) % self.nrots
                
                # Dos triángulos por cada segmento del grid
                faces.append((current, below, next_rot))
                faces.append((next_rot, below, below_next))

        # Reconvertir a numpy arrays para eficiencia
        faces = np.array(faces, dtype=np.int32)
        return vertices, faces

    def draw(self, img, rvec, tvec, K, dist, board_pos, square_size, override_color=None):
        col, row = board_pos
        offset_x = (col * square_size) + (square_size / 2)
        offset_y = (row * square_size) + (square_size / 2)
        
        # Copiamos los vértices para no modificar el original
        world_verts = self.vertices.copy()
        world_verts[:, 0] += offset_x
        world_verts[:, 1] += offset_y

        # Transformar a coordenadas de cámara
        R, _ = cv.Rodrigues(rvec)
        cam_verts_T = np.dot(R, world_verts.T) + tvec
        cam_verts = cam_verts_T.T

        # Proyectar a 2D
        imgpts_flat, _ = cv.projectPoints(world_verts, rvec, tvec, K, dist)
        imgpts = imgpts_flat.reshape(-1, 2).astype(np.int32)

        # Dibujar caras
        # Tenemos un array Nfaces x 3 x 3 (3 vértices por cara, cada uno con 3 coords)
        f_verts_3d = cam_verts[self.faces] 

        # Calcular normales y filtrar caras visibles
        edge1 = f_verts_3d[:, 1] - f_verts_3d[:, 0]
        edge2 = f_verts_3d[:, 2] - f_verts_3d[:, 0]
        normals = np.cross(edge1, edge2)
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms[norms == 0] = 1e-6
        normals /= norms

        # Ocultar caras que miran hacia atrás
        centers = np.mean(f_verts_3d, axis=1)
        culling_dot = np.einsum('ij,ij->i', normals, centers)
        visible_mask = culling_dot > 0
        # Aplicar máscara
        visible_faces = self.faces[visible_mask]
        visible_normals = normals[visible_mask]
        visible_z_centers = centers[visible_mask, 2]

        # Iluminación simple
        light_dir = np.array([0.5, -1.0, 0.5])
        light_dir /= np.linalg.norm(light_dir)
        diffuse = np.abs(np.dot(visible_normals, light_dir))
        intensity = 0.4 + 0.6 * diffuse
        intensity = np.clip(intensity, 0, 1)
        # Calcular los colores finales de las caras
        final_colors = (self.base_color * intensity[:, np.newaxis]).astype(np.uint8).tolist()

        # Ordenar las caras por profundidad (painter's algorithm)
        sorted_indices = np.argsort(visible_z_centers)[::-1]
        f_verts_2d = imgpts[visible_faces]

        # Dibujado
        for idx in sorted_indices:
            pts = f_verts_2d[idx]
            color = tuple(final_colors[idx])
            cv.fillConvexPoly(img, pts, color, lineType=cv.LINE_AA)


class Chessboard:
    def __init__(self, mtx, dist, square_size, assets_paths, nrots=12):
        """
        Args:
            mtx: Matriz intrínseca de la cámara.
            dist: Coeficientes de distorsión.
            square_size: Tamaño del lado de cada casilla en cm.
            assets_paths: Diccionario con rutas a perfiles de piezas, e.g. {'P': 'path/to/P.chsp'}
        """
        self.mtx = mtx
        self.dist = dist
        self.square_size = square_size
        self.nrots = nrots
        self.board = chess.Board()
        self.assets = self.load_assets(assets_paths)
        
    def load_assets(self, assets_paths):
        """
        Carga las piezas de ajedrez desde los archivos de perfil.
        
        Args:
            assets_paths: Diccionario con rutas a perfiles de piezas.
        """
        loaded_assets = {}

        for key, path in assets_paths.items():
            # Pieza Blanca
            loaded_assets[key.upper()] = ChessPiece(
                                            name=key,
                                            profile_path=path, 
                                            nrots=self.nrots, 
                                            color='white'
                                        )
    
            # Pieza Negra
            loaded_assets[key.lower()] = ChessPiece(
                                            name=key,
                                            profile_path=path, 
                                            nrots=self.nrots, 
                                            color='black'
                                        )
            #print(f"Cargada pieza {key} desde {path}")
        return loaded_assets
    
    def load_position(self, fen):
        """
        Carga una posición de ajedrez desde una cadena FEN.
        """
        try:
            self.board.set_fen(fen)
            print("Posición cargada desde FEN.")
        except ValueError:
            print("FEN inválida. Usando posición inicial.")
            self.board.reset()

    def draw(self, frame, rvec, tvec):
        """
        Dibuja el estado actual del tablero sobre el frame.
        Ordena las piezas por profundidad para correcto solapamiento.
        Args:
            frame: Imagen donde se dibujarán las piezas.
            rvec: Vector de rotación del tablero.
            tvec: Vector de traslación del tablero.
        """
        pieces_to_draw = []
        R, _ = cv.Rodrigues(rvec)

        # Recorremos las 64 casillas del tablero lógico
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            
            if piece:
                piece_type = piece.symbol()  # 'P', 'p', 'R', etc.
                if piece_type not in self.assets:
                    print(f"Advertencia: No se encontró el asset para la pieza '{piece_type}'")
                    continue 
                chess_piece = self.assets[piece_type]
                
                # Calcular posición de la casilla
                file_idx = chess.square_file(square) - 1  # 0 a 7 -> -1 a 6
                rank_idx = chess.square_rank(square) - 1  # 0 a 7 -> -1 a 6

                # Cambiar referencia a esquina superior izquierda
                visual_col = file_idx
                visual_row = 5 - rank_idx

                # Calcular profundidad para ordenamiento
                offset_x = (visual_col * self.square_size) + (self.square_size / 2)
                offset_y = (visual_row * self.square_size) + (self.square_size / 2)
                world_pos = np.array([[offset_x], [offset_y], [0.0]], dtype=np.float32)
                # Obtener posición en cámara
                cam_pos = np.dot(R, world_pos) + tvec
                depth_z = cam_pos[2, 0]

                # Guardar en lista para dibujar después
                pieces_to_draw.append({
                    'z': depth_z,
                    'piece': chess_piece,
                    'coords': (visual_col, visual_row)
                })

        # Ordenar piezas por profundidad (de mayor a menor z)
        pieces_to_draw.sort(key=lambda x: x['z'], reverse=True)

        # Dibujar piezas en orden
        for item in pieces_to_draw:
            item['piece'].draw(
                img=frame,
                rvec=rvec,
                tvec=tvec,
                K=self.mtx,
                dist=self.dist,
                board_pos=item['coords'],
                square_size=self.square_size
            )

    def reset_board(self):
        """
        Resetea el tablero a la posición inicial.
        """
        self.board.reset()

    def undo_move(self):
        """
        Deshace el último movimiento realizado.
        """
        if len(self.board.move_stack) > 0:
            move = self.board.pop()
            return True
        return False

    def make_move(self, move_uci):
        """
        Realiza un movimiento en el tablero si es legal.
        
        Args:
            move_uci: Movimiento en notación UCI (e.g., 'e2e4').
        
        Returns:
            bool: True si el movimiento fue realizado, False si fue ilegal.
        """
        try: 
            move = chess.Move.from_uci(move_uci)
            if move in self.board.legal_moves:
                self.board.push(move)
                return True
            else:
                print(f"Movimiento ilegal: {move_uci}")
                return False
        except ValueError:
            print(f"Formato de movimiento inválido: {move_uci}")
            return False
        
    def load_fen_from_file(self, filepath):
        """
        Carga una posición FEN desde un archivo de texto.
        
        Args:
            filepath: Ruta al archivo que contiene la cadena FEN.

        Raises:
            FileNotFoundError: Si el archivo no existe.
            ValueError: Si la cadena FEN es inválida.
        """
        try:
            with open(filepath, 'r') as f:
                fen = f.readline().strip()
                self.load_position(fen)
        except FileNotFoundError:
            raise
        except ValueError:
            raise