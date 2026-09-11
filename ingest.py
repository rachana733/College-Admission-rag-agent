import os
import tempfile
import ibm_boto3
from ibm_botocore.client import Config
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# ── IBM COS credentials from environment ──────────────────────────────────────
COS_API_KEY     = os.environ["COS_API_KEY"]
COS_INSTANCE_ID = os.environ["COS_INSTANCE_ID"]
COS_ENDPOINT    = os.environ["COS_ENDPOINT"]
COS_BUCKET      = os.environ["COS_BUCKET"]

# ── Connect to IBM Cloud Object Storage ───────────────────────────────────────
cos = ibm_boto3.client(
    "s3",
    ibm_api_key_id=COS_API_KEY,
    ibm_service_instance_id=COS_INSTANCE_ID,
    config=Config(signature_version="oauth"),
    endpoint_url=COS_ENDPOINT,
)

# ── Discover .txt objects in the bucket ───────────────────────────────────────
print(f"Listing .txt files in bucket '{COS_BUCKET}' ...")
response = cos.list_objects_v2(Bucket=COS_BUCKET)
objects  = response.get("Contents", [])
txt_keys = [obj["Key"] for obj in objects if obj["Key"].endswith(".txt")]

if not txt_keys:
    raise RuntimeError(f"No .txt files found in bucket '{COS_BUCKET}'.")

print(f"Found {len(txt_keys)} .txt file(s): {txt_keys}\n")

# ── Download each file, load, and split ───────────────────────────────────────
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
all_chunks = []

with tempfile.TemporaryDirectory() as tmpdir:
    for key in txt_keys:
        local_path = os.path.join(tmpdir, os.path.basename(key))

        print(f"  Downloading '{key}' ...", end=" ", flush=True)
        cos.download_file(COS_BUCKET, key, local_path)
        print("done.")

        loader = TextLoader(local_path, encoding="utf-8")
        docs   = loader.load()
        chunks = splitter.split_documents(docs)

        print(f"    → {len(docs)} document(s), {len(chunks)} chunk(s)")
        all_chunks.extend(chunks)

print(f"\nTotal chunks across all files: {len(all_chunks)}")

# ── Embed and store in ChromaDB ───────────────────────────────────────────────
print("\nLoading embedding model (all-MiniLM-L6-v2) ...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

print("Embedding chunks and persisting to ./chroma_db ...")
vectorstore = Chroma.from_documents(
    documents=all_chunks,
    embedding=embeddings,
    persist_directory="./chroma_db",
)

print(f"\nDone. {len(all_chunks)} chunk(s) stored in ./chroma_db")
