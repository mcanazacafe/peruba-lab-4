"""Servicio de analisis de imagenes con OpenCV."""
import cv2
import numpy as np


def analizar_imagen(path: str) -> dict:
    """
    Analiza una imagen y retorna sus dimensiones y si se detectaron bordes.

    Args:
        path: Ruta al archivo de imagen.

    Returns:
        Diccionario con alto, ancho y bordes_detectados.

    Raises:
        ValueError: Si la imagen no puede ser leida o es invalida.
    """
    imagen = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if imagen is None:
        raise ValueError(f"No se pudo leer la imagen en la ruta: {path}")

    if imagen.size == 0:
        raise ValueError("La imagen esta vacia.")

    bordes = cv2.Canny(imagen, 50, 150)

    return {
        "alto": int(imagen.shape[0]),
        "ancho": int(imagen.shape[1]),
        "bordes_detectados": int(bordes.sum() > 0),
    }


def crear_imagen_prueba(
    alto: int = 100,
    ancho: int = 100,
    color: int = 128,
    con_bordes: bool = False,
) -> np.ndarray:
    """
    Crea una imagen NumPy para pruebas.

    Args:
        alto: Altura de la imagen en pixeles.
        ancho: Ancho de la imagen en pixeles.
        color: Valor de gris de fondo (0-255).
        con_bordes: Si True, dibuja un rectangulo para generar bordes detectables.

    Returns:
        Imagen NumPy en escala de grises.
    """
    imagen = np.full((alto, ancho), color, dtype=np.uint8)
    if con_bordes:
        cv2.rectangle(imagen, (10, 10), (ancho - 10, alto - 10), 255, 2)
    return imagen
