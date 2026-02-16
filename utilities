import os
import re
from typing import List
import json
# from PIL import Image
from collections import defaultdict

# langchain imports
from datetime import datetime
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import Chroma

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

import fitz  # PyMuPDF

from config import llm
import textwrap
from collections import defaultdict


### image extraction
# ---------- Figure detection ----------
FIG_PATTERN = re.compile(
    r"\b(fig(?:ure)?)\s*[\.\-:–]?\s*(\d+(?:\.\s*\d+)*)\b",
    re.IGNORECASE
)

#-----------------------------------------------------------

def pretty_steps(text, width=100):
    steps = re.split(r'\s(?=\d+\.\s)', text.strip())
    output = []

    for step in steps:
        step = step.strip()
        if not step:
            continue
        wrapped = textwrap.fill(step, width=width)
        output.append(wrapped)

    return "\n\n".join(output)

#------------------------------------------------------------

def extract_figure_name(candidate_text: str):
    """
    Extract figure id in normalized format: 'Fig.2.1'
    """
    if not candidate_text:
        return None

    t = candidate_text.strip()
    m = FIG_PATTERN.search(t)
    if not m:
        return None

    num = m.group(2).replace(" ", "").strip(".")
    return f"Fig.{num}"

#----------------------------------------------------------------

# ---------- Main reusable function ----------
def extract_images_from_pdf(
    pdf_path: str,
    output_dir: str
):
    """
    Extract images from a PDF and return metadata.

    Returns:
        List[dict]: image records with paths and figure names
    """
    os.makedirs(output_dir, exist_ok=True)

    doc = fitz.open(pdf_path)
    image_records = []

    for page_index in range(len(doc)):
        page = doc[page_index]
        images = page.get_images(full=True)
        text_blocks = page.get_text("blocks")

        for img_index, img in enumerate(images, start=1):
            xref = img[0]
            image_name_in_pdf = img[7]  # e.g. 'Im12'

            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]

            image_filename = f"page_{page_index + 1}_img_{img_index}.{image_ext}"
            image_path = os.path.join(output_dir, image_filename)

            with open(image_path, "wb") as f:
                f.write(image_bytes)

            # ----- image bbox -----
            try:
                bbox = page.get_image_bbox(image_name_in_pdf)
            except Exception:
                bbox = None

            image_name = None
            context_before = ""
            context_after = ""

            if bbox:
                img_top = bbox.y0
                img_bottom = bbox.y1

                above_text = []
                below_text = []

                for block in text_blocks:
                    x0, y0, x1, y1, text, *_ = block
                    text = text.strip()
                    if not text:
                        continue

                    if y1 < img_top:
                        above_text.append((y1, text))
                    elif y0 > img_bottom:
                        below_text.append((y0, text))

                above_text.sort(key=lambda x: x[0], reverse=True)
                below_text.sort(key=lambda x: x[0])

                # ---- try nearest text first ----
                nearest_candidates = (
                    [t for _, t in above_text[:5]] +
                    [t for _, t in below_text[:5]]
                )

                for cand in nearest_candidates:
                    found = extract_figure_name(cand)
                    if found:
                        image_name = found
                        break

                # ---- fallback: whole page ----
                if image_name is None:
                    page_text = page.get_text("text")
                    image_name = extract_figure_name(page_text)

                # ---- build context ----
                for _, txt in above_text:
                    context_before += " " + txt
                    if len(context_before) >= 200:
                        break

                for _, txt in below_text:
                    context_after += " " + txt
                    if len(context_after) >= 200:
                        break

                if image_name is None:
                    image_name = (
                        extract_figure_name(context_before)
                        or extract_figure_name(context_after)
                    )

            record = {
                "source_pdf": pdf_path,
                "page": page_index + 1,
                "image_index": img_index,
                "image_path": image_path,
                "width": base_image.get("width"),
                "height": base_image.get("height"),
                "ext": image_ext,
                "image_name": image_name,
            }

            image_records.append(record)

    doc.close()
    return image_records

#---------------------------------------------------------------

def extract_figures_from_page_content(
    page_content: str,
    *,
    unique: bool = True,
    keep_order: bool = True,
    prefix: str = "Fig."
) -> List[str]:
    
    if not page_content:
        return []

    matches = FIG_PATTERN.findall(page_content)
    # matches: list of tuples like [("Fig", "19. 2"), ("Figure", "2.1"), ...]

    normalized = []
    for _, num in matches:
        num = num.replace(" ", "")      # "19. 2" -> "19.2"
        num = num.strip(".")            # "2.1." -> "2.1"
        normalized.append(f"{prefix}{num}")

    if not unique:
        return normalized

    # unique=True
    if keep_order:
        seen = set()
        out = []
        for f in normalized:
            if f not in seen:
                seen.add(f)
                out.append(f)
        return out

    return sorted(set(normalized))

#--------------------------------------------------

# def extract_pdf_name_from_metadata(metadata: dict) -> str:
#     """
#     Extract PDF filename from metadata['file_path'].

#     Expected format:
#       metadata['file_path'] = 'data/<pdf_name>'

#     Returns:
#       '<pdf_name>'  (e.g., 'demo2.pdf')
#     """
#     file_path = metadata.get("file_path")
#     if not file_path:
#         raise ValueError("metadata does not contain 'file_path'")

#     return os.path.basename(file_path)

def extract_pdf_name_from_metadata(metadata: dict) -> str:
    """
    Extract PDF filename from metadata['file_path'] without extension.

    Example:
      metadata['file_path'] = 'data/demo2.pdf'
      returns 'demo2'
    """
    file_path = metadata.get("file_path")
    if not file_path:
        raise ValueError("metadata does not contain 'file_path'")

    filename = os.path.basename(file_path)      # demo2.pdf
    pdf_name, _ = os.path.splitext(filename)    # ('demo2', '.pdf')
    return pdf_name

#-----------------------------------------------------------------------

# def get_figure_images(
#     pdf_name: str,
#     fig_name: str,
#     base_dir: str = "image_store",
#     show: bool = True,
#     print_paths: bool = True,
#     return_images: bool = False
# ):
#     import os, json
#     from PIL import Image

#     pdf_stem = os.path.splitext(pdf_name)[0]
#     json_path = os.path.join(base_dir, pdf_name, f"{pdf_stem}.json")

#     if not os.path.exists(json_path):
#         raise FileNotFoundError(
#             f"JSON mapping file not found at: {json_path}\n"
#             f"Expected structure: {base_dir}/{pdf_name}/{pdf_stem}.json"
#         )

#     with open(json_path, "r", encoding="utf-8") as f:
#         loaded_json = json.load(f)

#     image_paths = loaded_json.get(fig_name, [])

#     if print_paths:
#         if not image_paths:
#             print(f"No images found for {fig_name}.")
#             print("Available keys (sample):", list(loaded_json.keys())[:15])
#         else:
#             print(f"Found {len(image_paths)} images for {fig_name}")

#     images = []

#     if show and image_paths:
#         for p in image_paths:
#             try:
#                 img = Image.open(p)
#                 img.show()
#                 if return_images:
#                     images.append(img)
#             except Exception as e:
#                 print(f"Could not open {p}: {e}")

#     if return_images:
        # return images


def get_figure_images(
    pdf_name: str,
    fig_name: str,
    base_dir: str = "image_store",
    show: bool = True,
    print_paths: bool = True,
    return_images: bool = False,
    return_paths: bool = True   # ✅ NEW, default safe
):
    import os, json
    from PIL import Image

    # pdf_stem = os.path.splitext(pdf_name)[0]
    # json_path = os.path.join(base_dir, pdf_name, f"{pdf_stem}.json")

    # ✅ normalize: accept "demo2" or "demo2.pdf" or even "path/to/demo2.pdf"
    pdf_stem = os.path.splitext(os.path.basename(pdf_name))[0]

    # ✅ always look in image_store/<stem>/<stem>.json
    json_path = os.path.join(base_dir, pdf_stem, f"{pdf_stem}.json")


    if not os.path.exists(json_path):
        raise FileNotFoundError(
            f"JSON mapping file not found at: {json_path}\n"
            f"Expected structure: {base_dir}/{pdf_name}/{pdf_stem}.json"
        )

    with open(json_path, "r", encoding="utf-8") as f:
        loaded_json = json.load(f)

    image_paths = loaded_json.get(fig_name, [])

    if print_paths:
        if not image_paths:
            print(f"No images found for {fig_name}.")
            print("Available keys (sample):", list(loaded_json.keys())[:15])
        else:
            print(f"Found {len(image_paths)} images for {fig_name}")

    images = []

    if show and image_paths:
        for p in image_paths:
            try:
                img = Image.open(p)
                img.show()
                if return_images:
                    images.append(img)
            except Exception as e:
                print(f"Could not open {p}: {e}")

    # ✅ Fixed + explicit return contract
    if return_images:
        return images

    if return_paths:
        return image_paths

    return None
#-----------------------------------------------------------------------

# def get_all_images(docs):
#     fig_map = defaultdict(list)

#     for i, doc in enumerate(docs):
#         figs = extract_figures_from_page_content(doc.page_content)
#         pdf_name = extract_pdf_name_from_metadata(doc.metadata)    
#         fig_map[pdf_name] += figs

#     for pdf_name in fig_map.keys():
#         for fig_value in fig_map[pdf_name]:
#             get_figure_images(pdf_name, fig_value)


def get_all_images(docs):
    """
    Extract all figure images related to retrieved docs
    and return a list of image paths.
    """
    fig_map = defaultdict(set)
    image_paths = []

    for doc in docs:
        figs = extract_figures_from_page_content(doc.page_content)
        pdf_name = extract_pdf_name_from_metadata(doc.metadata)

        for fig in figs:
            fig_map[pdf_name].add(fig)

    for pdf_name, fig_values in fig_map.items():
        for fig_value in fig_values:
            paths = get_figure_images(
                pdf_name,
                fig_value,
                show=False,          # ✅ don't open images
                print_paths=False,   # ✅ don't print noise
                return_images=False,
                return_paths=True
                )
            if paths:
                image_paths.extend(paths)

    return image_paths

#-------------------------------------------------------------------------
### json map
def save_image_map_json(
    image_records,
    json_output_path,
    max_records=None,
    skip_missing_names=True
):
    """
    Create and save a JSON map:
      {
        "Fig.2.1": ["path1", "path2"],
        "Fig.3.4": ["path3"]
      }

    Args:
        image_records (list[dict]): output from extract_images_from_pdf
        json_output_path (str): path to .json file
        max_records (int | None): limit number of records processed
        skip_missing_names (bool): ignore records with image_name=None
    """

    os.makedirs(os.path.dirname(json_output_path), exist_ok=True)

    json_map = defaultdict(list)

    records = image_records[:max_records] if max_records else image_records

    for record in records:
        image_name = record.get("image_name")
        image_path = record.get("image_path")

        if skip_missing_names and not image_name:
            continue

        json_map[image_name].append(image_path)

    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(dict(json_map), f, indent=2, ensure_ascii=False)

    print(f"Saved JSON to {json_output_path}")

    return dict(json_map)

### pdf cleaning
# Join broken lines

def fix_line_breaks(text: str) -> str:
    # Join lines where a word is broken by newline
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    return text

# Fix hyphenated line breaks
def fix_hyphenation(text: str) -> str:
    # Remove hyphenation caused by line breaks
    text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)
    return text

# Remove page numbers & headers (targeted)
def remove_headers_footers(text: str) -> str:
    text = re.sub(r"Chapter\s+\d+.*?\n", "", text)
    text = re.sub(r"\bPage:\s*\d+\b", "", text)
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
    return text

# Combine everything
def clean_pdf_text(text: str) -> str:
    text = fix_hyphenation(text)
    text = fix_line_breaks(text)
    text = remove_headers_footers(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


## indexing the uploaded pdf
def index_pdf(pdf_path, db_path, original_filename=None):
    # ---------------- load pdf ----------------
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()

    # ----------- clean text + metadata --------
    ingestion_time = datetime.now().isoformat()
    source_basename = os.path.basename(original_filename or pdf_path)

    for doc in documents:
        doc.page_content = clean_pdf_text(doc.page_content)
        doc.metadata["ingestion_time"] = ingestion_time
        doc.metadata["type"] = "text"

        # if original_filename:
        #     doc.metadata["source"] = original_filename

        doc.metadata["source"] = source_basename
        doc.metadata["source_path"] = (original_filename or pdf_path).replace("\\", "/")

    # ---------------- split -------------------
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
        separators=["\n\n", "\n", " "]
    )

    chunks = text_splitter.split_documents(documents)

    # ---------------- embeddings --------------
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        # model_kwargs={"device": "cuda"}  # optional
    )

    # -------- create & persist vector store ---
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=db_path,
        collection_name="rag_collection"
    )

    # vectorstore.persist()

    print(f"Vector store created with {vectorstore._collection.count()} vectors")
    print(f"Persisted to: {db_path}")

    return vectorstore

## sementic search from the vector store
def semantic_search(query, db):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = Chroma(
        embedding_function=embeddings,
        persist_directory=db,
        collection_name="rag_collection"
    )

    # ✅ NEW: check if vectorstore has any chunks
    try:
        chunk_count = vector_store._collection.count()
    except Exception:
        chunk_count = 0  # safest fallback

    if chunk_count == 0:
        return [], "⚠️ Vector store is empty. Please index a PDF first."

    retriever = vector_store.as_retriever(
        search_kwargs={"k": 5}
    )

    raw_retriever_output = retriever.invoke(query)

    system_prompt = """You are an assistant for question-answering tasks.
    Use only the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say that you don't know.
    Use three sentences maximum and keep the answer concise
    you can explain in detail if the query says so.

    Context:
    {context}
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}")
    ])

    rag_chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
    )

    response = rag_chain.invoke(query)
    response_llm = pretty_steps(response.content, width=120)
    return raw_retriever_output, response_llm

#---------------list out pdfs-----------------------

def list_indexed_pdfs(db_path):
    """
    Returns a sorted list of unique PDF filenames
    present in the vector store (true source metadata).
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = Chroma(
        persist_directory=db_path,
        embedding_function=embeddings,
        collection_name="rag_collection"
    )

    data = vector_store._collection.get(include=["metadatas"])

    pdf_names = set()

    for meta in data["metadatas"]:
        if meta and "source" in meta and meta["source"]:
            pdf_names.add(os.path.basename(meta["source"]))

    return sorted(pdf_names)


def delete_pdf_from_chroma(pdf_name: str, db_path: str):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = Chroma(
        persist_directory=db_path,
        embedding_function=embeddings,
        collection_name="rag_collection"
    )

    # ✅ SINGLE filter key (this is the key point)
    vector_store._collection.delete(
        where={"source": pdf_name}
    )

    try:
        vector_store.persist()
    except Exception:
        pass

    print(f"✅ Deleted all chunks for: {pdf_name}")

#-----------------------------------------------------------
# inspect metadata

def debug_print_sample_metadata(db_path: str, n: int = 1, collection_name: str = "rag_collection"):
    """
    Prints metadata (and ids returned by default) of the first n chunks
    in the Chroma collection.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = Chroma(
        persist_directory=db_path,
        embedding_function=embeddings,
        collection_name=collection_name
    )

    # ✅ ids are returned by default; don't put "ids" in include
    data = vector_store._collection.get(include=["metadatas"], limit=n)

    if not data or not data.get("ids"):
        print("⚠️ No chunks found in the collection.")
        return

    for i, (_id, meta) in enumerate(zip(data["ids"], data["metadatas"]), start=1):
        print(f"\n--- Chunk {i} ---")
        print("id:", _id)
        print("metadata:", meta)

