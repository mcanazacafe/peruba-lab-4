"""
Pruebas unitarias e integracion para la API de Vision Computacional.
Cubre: image_service y endpoints FastAPI.
"""
import io

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from lab4_api_cv.api.main import app
from lab4_api_cv.services.image_service import analizar_imagen, crear_imagen_prueba

client = TestClient(app)


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────


@pytest.fixture
def imagen_con_bordes_path(tmp_path):
    """Genera un archivo PNG con bordes detectables y retorna su ruta."""
    img = crear_imagen_prueba(100, 100, color=50, con_bordes=True)
    path = str(tmp_path / "con_bordes.png")
    cv2.imwrite(path, img)
    return path


@pytest.fixture
def imagen_sin_bordes_path(tmp_path):
    """Genera un archivo PNG uniforme (sin bordes) y retorna su ruta."""
    img = crear_imagen_prueba(100, 100, color=128, con_bordes=False)
    path = str(tmp_path / "sin_bordes.png")
    cv2.imwrite(path, img)
    return path


def _imagen_bytes(alto=100, ancho=100, con_bordes=False) -> bytes:
    """Retorna una imagen PNG en memoria como bytes."""
    img = crear_imagen_prueba(alto, ancho, con_bordes=con_bordes)
    success, buf = cv2.imencode(".png", img)
    assert success
    return buf.tobytes()


# ──────────────────────────────────────────────
# Tests – image_service
# ──────────────────────────────────────────────


class TestAnalizarImagen:
    """Pruebas unitarias para la funcion analizar_imagen."""

    def test_retorna_dimensiones_correctas(self, imagen_con_bordes_path):
        resultado = analizar_imagen(imagen_con_bordes_path)
        assert resultado["alto"] == 100
        assert resultado["ancho"] == 100

    def test_detecta_bordes_cuando_existen(self, imagen_con_bordes_path):
        resultado = analizar_imagen(imagen_con_bordes_path)
        assert resultado["bordes_detectados"] == 1

    def test_no_detecta_bordes_en_imagen_uniforme(self, imagen_sin_bordes_path):
        resultado = analizar_imagen(imagen_sin_bordes_path)
        assert resultado["bordes_detectados"] == 0

    def test_resultado_contiene_claves_esperadas(self, imagen_con_bordes_path):
        resultado = analizar_imagen(imagen_con_bordes_path)
        assert "alto" in resultado
        assert "ancho" in resultado
        assert "bordes_detectados" in resultado

    def test_bordes_detectados_es_entero(self, imagen_con_bordes_path):
        resultado = analizar_imagen(imagen_con_bordes_path)
        assert isinstance(resultado["bordes_detectados"], int)
        assert resultado["bordes_detectados"] in (0, 1)

    def test_error_ruta_invalida(self):
        with pytest.raises(ValueError, match="No se pudo leer"):
            analizar_imagen("ruta/que/no/existe.png")

    def test_imagen_rectangular(self, tmp_path):
        img = crear_imagen_prueba(alto=200, ancho=300, con_bordes=True)
        path = str(tmp_path / "rect.png")
        cv2.imwrite(path, img)
        resultado = analizar_imagen(path)
        assert resultado["alto"] == 200
        assert resultado["ancho"] == 300

    def test_imagen_pequena(self, tmp_path):
        img = crear_imagen_prueba(alto=10, ancho=10, con_bordes=False)
        path = str(tmp_path / "small.png")
        cv2.imwrite(path, img)
        resultado = analizar_imagen(path)
        assert resultado["alto"] == 10
        assert resultado["ancho"] == 10


class TestCrearImagenPrueba:
    """Pruebas para la funcion auxiliar crear_imagen_prueba."""

    def test_dimensiones(self):
        img = crear_imagen_prueba(80, 60)
        assert img.shape == (80, 60)

    def test_color_fondo(self):
        img = crear_imagen_prueba(50, 50, color=200)
        assert img[25, 25] == 200

    def test_con_bordes_modifica_imagen(self):
        sin = crear_imagen_prueba(100, 100, con_bordes=False)
        con = crear_imagen_prueba(100, 100, con_bordes=True)
        assert not np.array_equal(sin, con)


# ──────────────────────────────────────────────
# Tests – endpoint GET /
# ──────────────────────────────────────────────


class TestHomeEndpoint:
    """Pruebas para el endpoint GET /."""

    def test_status_200(self):
        response = client.get("/")
        assert response.status_code == 200

    def test_mensaje_correcto(self):
        response = client.get("/")
        assert response.json() == {"mensaje": "API de Visión Computacional funcionando"}

    def test_content_type_json(self):
        response = client.get("/")
        assert "application/json" in response.headers["content-type"]


# ──────────────────────────────────────────────
# Tests – endpoint POST /analyze-image
# ──────────────────────────────────────────────


class TestAnalyzeImageEndpoint:
    """Pruebas de integracion para POST /analyze-image."""

    def test_imagen_valida_retorna_200(self):
        data = _imagen_bytes(con_bordes=True)
        response = client.post(
            "/analyze-image",
            files={"file": ("test.png", io.BytesIO(data), "image/png")},
        )
        assert response.status_code == 200

    def test_respuesta_tiene_claves_esperadas(self):
        data = _imagen_bytes(con_bordes=True)
        response = client.post(
            "/analyze-image",
            files={"file": ("test.png", io.BytesIO(data), "image/png")},
        )
        body = response.json()
        assert "mensaje" in body
        assert "resultado" in body
        resultado = body["resultado"]
        assert "alto" in resultado
        assert "ancho" in resultado
        assert "bordes_detectados" in resultado

    def test_mensaje_procesamiento_exitoso(self):
        data = _imagen_bytes(con_bordes=True)
        response = client.post(
            "/analyze-image",
            files={"file": ("test.png", io.BytesIO(data), "image/png")},
        )
        assert response.json()["mensaje"] == "Procesamiento exitoso"

    def test_dimensiones_correctas_en_respuesta(self):
        data = _imagen_bytes(alto=50, ancho=80, con_bordes=False)
        response = client.post(
            "/analyze-image",
            files={"file": ("test.png", io.BytesIO(data), "image/png")},
        )
        resultado = response.json()["resultado"]
        assert resultado["alto"] == 50
        assert resultado["ancho"] == 80

    def test_detecta_bordes_en_imagen_con_bordes(self):
        data = _imagen_bytes(con_bordes=True)
        response = client.post(
            "/analyze-image",
            files={"file": ("test.png", io.BytesIO(data), "image/png")},
        )
        assert response.json()["resultado"]["bordes_detectados"] == 1

    def test_sin_bordes_en_imagen_uniforme(self):
        data = _imagen_bytes(con_bordes=False)
        response = client.post(
            "/analyze-image",
            files={"file": ("test.png", io.BytesIO(data), "image/png")},
        )
        assert response.json()["resultado"]["bordes_detectados"] == 0

    def test_extension_no_permitida_retorna_400(self):
        response = client.post(
            "/analyze-image",
            files={"file": ("malware.exe", io.BytesIO(b"fake"), "application/octet-stream")},
        )
        assert response.status_code == 400

    def test_extension_txt_no_permitida(self):
        response = client.post(
            "/analyze-image",
            files={"file": ("archivo.txt", io.BytesIO(b"texto"), "text/plain")},
        )
        assert response.status_code == 400

    def test_extension_pdf_no_permitida(self):
        response = client.post(
            "/analyze-image",
            files={"file": ("doc.pdf", io.BytesIO(b"%PDF"), "application/pdf")},
        )
        assert response.status_code == 400

    def test_archivo_corrupto_retorna_422(self):
        response = client.post(
            "/analyze-image",
            files={"file": ("corrupta.png", io.BytesIO(b"no-es-imagen"), "image/png")},
        )
        assert response.status_code == 422

    def test_imagen_jpg_aceptada(self):
        img = crear_imagen_prueba(60, 60, con_bordes=True)
        _, buf = cv2.imencode(".jpg", img)
        data = buf.tobytes()
        response = client.post(
            "/analyze-image",
            files={"file": ("foto.jpg", io.BytesIO(data), "image/jpeg")},
        )
        assert response.status_code == 200

    def test_imagen_bmp_aceptada(self):
        img = crear_imagen_prueba(60, 60, con_bordes=False)
        _, buf = cv2.imencode(".bmp", img)
        data = buf.tobytes()
        response = client.post(
            "/analyze-image",
            files={"file": ("foto.bmp", io.BytesIO(data), "image/bmp")},
        )
        assert response.status_code == 200
