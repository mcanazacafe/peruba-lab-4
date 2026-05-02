"""API REST de Vision Computacional - Endpoints FastAPI."""
import logging
import os
import shutil
import uuid


from fastapi import FastAPI, HTTPException, UploadFile, status

from lab4_api_cv.services.image_service import analizar_imagen

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="API de Vision Computacional",
    description="API REST para analisis de imagenes con deteccion de bordes (OpenCV).",
    version="1.0.0",
)

# Directorio seguro para subidas, resuelto a una ruta absoluta
_BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
UPLOAD_DIR = os.path.join(_BASE_DIR, "data")
ALLOWED_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"})
os.makedirs(UPLOAD_DIR, exist_ok=True)


def _ruta_segura(directorio: str, nombre_archivo: str) -> str:
    """
    Construye una ruta segura dentro de `directorio`, previniendo path traversal.
    Rechaza nombres con separadores, segmentos relativos o que escapen del directorio.
    """
    ruta = os.path.abspath(os.path.join(directorio, nombre_archivo))
    if not ruta.startswith(os.path.abspath(directorio) + os.sep):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Ruta de archivo invalida.",
        )
    return ruta


@app.get("/", tags=["Health"])
def home() -> dict:
    """Endpoint de salud para verificar que la API esta funcionando."""
    return {"mensaje": "API de Visión Computacional funcionando"}


@app.post("/analyze-image", tags=["Analisis"])
def analyze_image(file: UploadFile) -> dict:
    """
    Analiza una imagen subida y devuelve sus dimensiones y si se detectaron bordes.

    - **file**: Archivo de imagen (jpg, jpeg, png, bmp, tiff, webp)
    """
    nombre_original = file.filename or ""
    extension = os.path.splitext(nombre_original)[-1].lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Tipo de archivo no permitido '{extension}'. "
                f"Permitidos: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            ),
        )

    # Generar nombre seguro con UUID para evitar inyeccion de ruta y colisiones
    nombre_seguro = f"{uuid.uuid4().hex}{extension}"
    ruta = _ruta_segura(UPLOAD_DIR, nombre_seguro)

    try:
        with open(ruta, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info("Imagen guardada de forma segura.")

        resultado = analizar_imagen(ruta)
        logger.info("Analisis completado.")

        return {"mensaje": "Procesamiento exitoso", "resultado": resultado}

    except ValueError as exc:
        logger.error("Error de validacion al procesar la imagen.")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    except OSError as exc:
        logger.error("Error de E/S al procesar la imagen.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error interno al procesar la imagen.",
        ) from exc

    finally:
        if os.path.exists(ruta):
            os.remove(ruta)
            logger.info("Archivo temporal eliminado.")
