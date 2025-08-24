import sys
import yaml
from pathlib import Path
from typing import List
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from abc import ABC, abstractmethod
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from src.utils.utility import load_yaml

from typing import List

default_chunking_config = {"chunk_size": 512, "chunk_overlap": 50}
cfg = load_yaml("config.yaml")
chunking_config = cfg.get("chunking", default_chunking_config)


class BaseDocumentLoader(ABC):
    """Abstract base class for document loaders."""

    def __init__(self) -> None:
        super().__init__()
        self.CHUNK_SIZE = chunking_config["chunk_size"]
        self.CHUNK_OVERLAP = chunking_config["chunk_overlap"]

    @abstractmethod
    def load(self) -> List[Document]:
        """Load documents and return a list of LangChain Documents."""
        pass

    def load_and_split(self) -> List[Document]:
        """Load documents and split them into smaller chunks."""
        docs = self.load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.CHUNK_SIZE,
            chunk_overlap=self.CHUNK_OVERLAP,
            add_start_index=True,  # useful to track original positions
        )
        return splitter.split_documents(docs)


class PDFLoader(BaseDocumentLoader):
    """Loader for PDF files."""

    def __init__(self, file_path: str):
        super().__init__()
        self.file_path = file_path

    def load(self) -> List[Document]:
        loader = PyPDFLoader(self.file_path)
        return loader.load()


class TXTLoader(BaseDocumentLoader):
    """Loader for TXT files."""

    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> List[Document]:
        loader = TextLoader(self.file_path)
        return loader.load()


if __name__ == "__main__":
    pdf_loader = PDFLoader("data/raw/mycv_2025.pdf")
    pdf_chunks = pdf_loader.load_and_split()

    print(f"Loaded {len(pdf_chunks)} chunks from PDF")
    print("First chunk preview:\n", pdf_chunks[0].page_content[:200])
