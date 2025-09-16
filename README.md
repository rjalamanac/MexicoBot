# Contract Bot

Contract Bot is a Python-based tool that allows you to query PDF contracts using embeddings and a vector store. It supports building a local data store from PDFs and running an interactive chatbot to answer questions.


## Features

- Extract text from PDF documents.
- Build embeddings and store them in a vector store (`.pkl` file).
- Query your PDFs using a chatbot interface.
- Dockerized for easy deployment.



## Requirements

- Docker
- Python 3.10+ (if running locally)
- Tesseract OCR (for scanned PDFs, optional)
- Python dependencies (`requirements.txt` provided)


## Setup

### Using Docker

Build the Docker image:

```bash
docker build -t contract-bot:latest .
````

Run the container to **build the embeddings and store them**:

```bash
docker run --rm -it \
  -v ${PWD}/pdfs:/app/pdfs \
  -v ${PWD}:/app \
  contract-bot \
  python contract_bot.py --build --pdf_folder ./pdfs --datafile ./contract_store.pkl
```

* `pdfs` folder contains the PDF documents.
* `contract_store.pkl` will store the embeddings and vector data.

Run the container to **start the interactive chatbot**:

```bash
docker run --rm -it \
  -v ${PWD}/pdfs:/app/pdfs \
  -v ${PWD}:/app \
  contract-bot
```

> Make sure the `contract_store.pkl` file exists before running the chatbot.



### Running Locally (without Docker)

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Place your PDF documents in a folder (e.g., `pdfs/`).

3. Build the vector store:

```bash
python contract_bot.py --build --pdf_folder ./pdfs --datafile ./contract_store.pkl
```

4. Run the chatbot:

```bash
python contract_bot.py
```



## Configuration

* `--pdf_folder`: Folder containing your PDF documents.
* `--datafile`: Pickle file where embeddings/vector store will be saved.
* Optional Tesseract OCR can be configured for scanned PDFs.



## Notes

* The first run should always include the `--build` step to generate the vector store.
* Re-run the build step whenever you add or update PDFs.
* Docker volumes ensure that your PDFs and data store persist outside the container.

