import json
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding

PROCESSED_DIR = Path("./corpus/processed")

def build_hierarchical_index():
    # Initialize connection to local Qdrant instance
    client = QdrantClient(host="localhost", port=6333)
    
    # Configure collection for standard Nomic Embed text dimensions (768)
    if not client.collection_exists("academic_corpus"):
        client.create_collection(
            collection_name="academic_corpus",
            vectors_config=VectorParams(size=768, distance=Distance.COSINE)
        )
        
    vector_store = QdrantVectorStore(client=client, collection_name="academic_corpus")
    docstore = SimpleDocumentStore()
    
    # Instantiate embedding configuration natively using local Ollama engine
    embed_model = OllamaEmbedding(model_name="nomic-embed-text", base_url="http://localhost:11434")
    
    llama_docs = []
    
    # Process structured intermediate outputs
    for json_file in PROCESSED_DIR.glob("*.json"):
        with open(json_file, "r") as f:
            paragraphs = json.load(f)
            
        for p in paragraphs:
            # Build LlamaIndex document object injecting deterministic citations directly into payloads
            doc = Document(
                text=p["text"],
                metadata={
                    "title": p["metadata"]["title"],
                    "author": p["metadata"]["author"],
                    "year": p["metadata"]["year"],
                    "page": p["page"],
                    "paragraph": p["paragraph_index"]
                }
            )
            llama_docs.append(doc)

    # Segment documents into Parent (2048), Mid (512), and Child Leaf (128) layers
    node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=[2048, 512, 128])
    nodes = node_parser.get_nodes_from_documents(llama_docs)
    leaf_nodes = get_leaf_nodes(nodes)
    
    # Store relationship mappings in docstore
    docstore.add_documents(nodes)
    
    storage_context = StorageContext.from_defaults(vector_store=vector_store, docstore=docstore)
    
    print(f"Indexing {len(leaf_nodes)} leaf child units into Qdrant store...")
    index = VectorStoreIndex(
        leaf_nodes,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True
    )
    
    # Persist mapping relationships locally
    docstore.persist("./index_storage/docstore.json")
    print("Vector storage indexing pass complete.")

if __name__ == "__main__":
    build_hierarchical_index()