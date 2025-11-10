from google import genai
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from src.Math.entity.config_entity import DataIngestion
from src.Math import logger
from pathlib import Path


class DataLoader:
    def __init__(self, config: DataIngestion):
        self.config = config
        self.splitter = SentenceSplitter(chunk_size=100, chunk_overlap=20)
        self.client = genai.Client()
        self._query_cache = {}  # Initialize in-memory cache for query embeddings

    def load_and_chunk_pdf(self, filename: Path):
        filename = self.config.docs / filename.name
        logger.info(filename)
        docs = PDFReader().load_data(filename)
        texts = [d.text for d in docs if getattr(d, "text", None)]
        chunks = []
        for t in texts:
            chunks.extend(self.splitter.split_text(t))
        return chunks

    def embed_texts(self, texts: list[str]):
        response = self.client.models.embed_content(
            model=self.config.EMBED_MODEL,
            contents=texts,
            config={
                "task_type": "RETRIEVAL_DOCUMENT",
            },
        )
        return [item.values for item in response.embeddings]

    def embed_query(self, query: str):
        # Check cache first
        if query in self._query_cache:
            logger.info(f"Query embedding cache hit for: '{query[:50]}...'")
            return self._query_cache[query]

        logger.info(f"Query embedding cache miss. Embedding: '{query[:50]}...'")
        response = self.client.models.embed_content(
            model=self.config.EMBED_MODEL,
            contents=[query],
            config={
                "task_type": "RETRIEVAL_QUERY",
            },
        )
        embedding = response.embeddings[0].values
        # Store in cache
        self._query_cache[query] = embedding
        return embedding
