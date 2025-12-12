# Detección Genial de Señales de Tráfico (DGST)

Proyecto de detección de señales de tráfico utilizando técnicas de visión artificial y aprendizaje automático.

## Autores

- Juan Diego Gallego Nicolás
- Óscar Vera López

## Requisitos

- Python 3.12.3 o superior.
- Dependencias listadas en `requirements.txt`.
- Make (para compilar librerías C).
- Clang (para compilar librerías C).
- OpenMP (versión de Clang para desarrollo). En Ubuntu: `libomp-dev`.
- (Opcional) UV para gestión de entornos y dependencias.

> ![IMPORTANT]
> Si ocurren errores de ejecución relacionados con la librería en C, es posible que tu sistema no soporte
> instrucciones SIMD. En ese caso, edita el Makefile en `src/dgst/filters/ffi/Makefile` y elimina las banderas
> relacionadas con SIMD.


Uso rápido:
 
 - (Opcional) Crear y activar un entorno virtual:
```bash
# Con venv:
python -m venv .venv
source .venv/bin/activate 
# Con UV:
uv new dgst_env
source .venv/bin/activate
```

 - (Opcional) Inicializar UV si se tiene instalado:
 
 ```bash
 uv sync
 ```

 - Instalar dependencias:
```bash
# Con pip:
pip install -r requirements.txt
# Con UV:
uv pip install -r requirements.txt
```

 - (Opcional) Compilar librerias C.
```bash
cd src/dgst/filters/ffi
make
```

 - Instalar el paquete DGST en el entorno:
```bash
# Con pip:
pip install -e .
# Con UV:
uv pip install -e .
```
