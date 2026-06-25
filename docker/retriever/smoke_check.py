import importlib.metadata as metadata
import subprocess

import datasets
import faiss
import fastapi
import pyserini
import torch
import transformers
import uvicorn
from pyserini.search.lucene import LuceneSearcher
from search_r1.search import retrieval_server


def version(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "unknown"


java = subprocess.run(
    ["java", "-version"],
    check=True,
    capture_output=True,
    text=True,
)

print("python retriever smoke check")
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("torchvision", version("torchvision"))
print("torchaudio", version("torchaudio"))
print("faiss", getattr(faiss, "__version__", "unknown"))
print("faiss gpu api", hasattr(faiss, "StandardGpuResources"))
print("transformers", transformers.__version__)
print("datasets", datasets.__version__)
print("pyserini", version("pyserini"))
print("pyserini module", pyserini.__name__)
print("lucene searcher", LuceneSearcher.__name__)
print("retrieval server", retrieval_server.DenseRetriever.__name__)
print("fastapi", fastapi.__version__)
print("uvicorn", uvicorn.__version__)
print("java", java.stderr.splitlines()[0])
print("retriever smoke check ok")
