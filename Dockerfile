FROM python:3.12-slim-bookworm

# Instalar dependencias del sistema: poppler + tesseract + español
RUN apt-get update && apt-get install -y \
    build-essential \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-spa \
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    libtiff-dev \
    libwebp-dev \
    && rm -rf /var/lib/apt/lists/*


# Crear directorio de trabajo
WORKDIR /app

# Copiar requirements
COPY requirements.txt .

# Instalar dependencias Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el bot
COPY contract_bot.py .
COPY pdfs ./pdfs

# Punto de entrada por defecto
CMD ["python", "contract_bot.py", "--ask", "--datafile", "./contract_store.pkl"]
