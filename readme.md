# CTrag

<p align="center">
    <img src="images/logo.webp" width="400s">
</p>

CTrag is a cyber threat intelligence tool that uses local large language models (LLMs) and a vector database to answer your questions about cyber threats. It's built on top of Langchain, Ollama, Chroma, and PyPDF.

## Credits

The majority of the code for this project was adapted from the work of the first developer of local-LLM-with-RAG. The project was then adapted for the specific cyber threat intelligence use-case and a first vector database was built.

https://github.com/amscotti/local-LLM-with-RAG

## Requirements

- [Ollama](https://ollama.ai/) version 0.1.26 or higher.
  - You can download other models with `ollama pull [MODEL_NAME]` so that the chatbot can use them (default: dolphin-mistral)

## Setup

1. Clone this repository to your local machine using Git LFS to ensure you receive the already processed vector database with the 2200 reports. If you haven't installed Git LFS, please follow the instructions [here](https://git-lfs.github.com/).
2. Create a Python virtual environment by running `python3 -m venv env`.
3. Activate the virtual environment by running `source env/bin/activate` on Unix or MacOS, or `.\env\Scripts\activate` on Windows.
4. Install the required Python packages by running `pip install -r requirements.txt`.

## Running the project

**Note:** The first time you run the project, it will download the necessary models from Ollama for the LLM and embeddings. This is a one-time setup process and may take some time depending on your internet connection.

1. Ensure your virtual environment is activated.
2. Run the streamlit GUI with `python streamlit run app.py`

## Adding reports to the vector database

1. Put the PDFs in the `reports` folder
2. Run the tool, it will automatically process the files and add them to the db.

## Source used to build the original vector database

- [VX-Underground archives](https://vx-underground.org/)

## Technologies Used

- [Langchain](https://github.com/langchain/langchain): A Python library for working with Large Language Model
- [Ollama](https://ollama.ai/): A platform for running Large Language models locally.
- [Chroma](https://docs.trychroma.com/): A vector database for storing and retrieving embeddings.
- [PyPDF](https://pypi.org/project/PyPDF2/): A Python library for reading and manipulating PDF files.

## License

This project is licensed under the terms of the MIT license.