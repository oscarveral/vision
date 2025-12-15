import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from pathlib import Path
import argparse

# Ruta a la carpeta del dataset procesado
DATASET_DIR = Path(__file__).parents[2] / "notebooks" / "images" / "bt4" 

# ID de la imagen. Parsear
parser = argparse.ArgumentParser(description="Visualizador de bounding boxes YOLOv8")
parser.add_argument(
    "--frame_id", 
    type=str, 
    required=True, 
    help="ID de la imagen a visualizar (sin extensión)"
)
args = parser.parse_args()
FRAME_ID_A_VISUALIZAR = args.frame_id


# Mapeo inverso para mostrar nombres (0 -> TrafficSign)
CLASS_NAMES = {
    0: "TrafficSign"
}

def buscar_archivo(dataset_dir, tipo, frame_id, extension):
    """
    Busca el archivo en las subcarpetas 'train' y 'val'.
    tipo: 'images' o 'labels'
    """
    splits = ['train', 'val']
    for split in splits:
        path = dataset_dir / tipo / split / f"{frame_id}{extension}"
        if path.exists():
            return path, split
    return None, None

def visualizar_yolo(frame_id):
    print(f"--- Buscando {frame_id} en {DATASET_DIR} ---")
    
    # Buscar Imagen
    img_path, split = buscar_archivo(DATASET_DIR, "images", frame_id, ".jpg")
    
    if not img_path:
        print(f"Error: No se encontró la imagen {frame_id}.jpg en train ni en val.")
        return

    print(f"Imagen encontrada en: {split}")

    # Cargar Imagen
    try:
        im = Image.open(img_path)
        img_w, img_h = im.size
    except Exception as e:
        print(f"Error abriendo imagen: {e}")
        return

    # Buscar Etiquetas
    txt_path, _ = buscar_archivo(DATASET_DIR, "labels", frame_id, ".txt")
    
    detecciones = []
    if txt_path:
        print(f"Etiquetas encontradas: {txt_path.name}")
        with open(txt_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls_id = int(parts[0])
                    xc = float(parts[1])
                    yc = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])
                    detecciones.append((cls_id, xc, yc, w, h))
    else:
        print("Nota: No hay archivo .txt (es una imagen de background/vacía).")

    # Visualizar
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.imshow(im)
    ax.set_title(f"ID: {frame_id} ({split}) - {len(detecciones)} objetos", fontsize=14)

    # Dibujar Cajas
    for cls_id, xc, yc, w, h in detecciones:        
        box_w = w * img_w
        box_h = h * img_h
        x_min = (xc * img_w) - (box_w / 2)
        y_min = (yc * img_h) - (box_h / 2)
        
        # Crear rectángulo
        rect = patches.Rectangle(
            (x_min, y_min), 
            box_w, 
            box_h, 
            linewidth=2, 
            edgecolor='#00ff00', # Verde brillante
            facecolor='none'
        )
        ax.add_patch(rect)
        
        # Etiqueta
        # label_name = CLASS_NAMES.get(cls_id, str(cls_id))
        # ax.text(
        #     x_min, y_min - 5, 
        #     label_name, 
        #     color='black', 
        #     fontsize=9, 
        #     backgroundcolor='#00ff00'
        # )

    plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualizar_yolo(FRAME_ID_A_VISUALIZAR)