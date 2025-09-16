# Contract Bot

Contract Bot es una herramienta en Python que permite consultar contratos en PDF usando embeddings y un vector store. Soporta la construcción de una base de datos local a partir de PDFs y un chatbot interactivo para responder preguntas.


## Características

* Extracción de texto de documentos PDF.
* Construcción de embeddings y almacenamiento en un vector store (`.pkl`).
* Consulta de PDFs mediante una interfaz de chatbot.
* Extracción dinámica de entidades clave desde un archivo externo (`ImportantWords.txt`).
* Preprocesamiento basado en secciones semánticas usando palabras importantes definidas por el usuario.
* Dockerizado para fácil despliegue.


## Requisitos

* Docker
* Python 3.10+ (si se ejecuta localmente)
* Tesseract OCR (para PDFs escaneados, opcional)
* Dependencias Python (`requirements.txt` proporcionado)

## Configuración

### Archivo `ImportantWords.txt`

Para personalizar la extracción de entidades y el preprocesamiento, crea un archivo llamado `ImportantWords.txt` en la raíz del proyecto, con una palabra clave por línea, por ejemplo:

```
Contrato
Cláusula
Artículo
```

El sistema leerá estas palabras para identificar secciones importantes y para guiar la división semántica del texto.


## Uso

### Usando Docker

Construir la imagen Docker:

```bash
docker build -t contract-bot:latest .
```

Ejecutar el contenedor para **construir los embeddings y almacenar los datos**:

```bash
docker run --rm -it \
  -v ${PWD}/pdfs:/app/pdfs \
  -v ${PWD}:/app \
  contract-bot \
  python contract_bot.py --build --pdf_folder ./pdfs --datafile ./contract_store.pkl
```

* La carpeta `pdfs` contiene los documentos PDF.
* El archivo `contract_store.pkl` almacenará los embeddings y datos vectoriales.

Ejecutar el contenedor para **iniciar el chatbot interactivo**:

```bash
docker run --rm -it \
  -v ${PWD}/pdfs:/app/pdfs \
  -v ${PWD}:/app \
  contract-bot \
  python contract_bot.py --ask
```

> Asegúrate de que `contract_store.pkl` exista antes de ejecutar el chatbot.


### Ejecución local (sin Docker)

1. Instalar dependencias:

```bash
pip install -r requirements.txt
```

2. Colocar tus PDFs en una carpeta (ejemplo: `pdfs/`).

3. Crear la base de datos vectorial:

```bash
python contract_bot.py --build --pdf_folder ./pdfs --datafile ./contract_store.pkl
```

4. Ejecutar el chatbot:

```bash
python contract_bot.py --ask
```


## Opciones de configuración

* `--pdf_folder`: Carpeta con documentos PDF.
* `--datafile`: Archivo pickle donde se guardarán embeddings y vectores.
* `ImportantWords.txt`: Archivo con las palabras clave para extracción y segmentación.
* OCR Tesseract configurable para PDFs escaneados.

## Notas

* La primera ejecución debe incluir `--build` para generar la base de datos.
* Repite el paso de build cada vez que agregues o modifiques PDFs.
* Los volúmenes Docker aseguran que tus PDFs y datos persistan fuera del contenedor.

