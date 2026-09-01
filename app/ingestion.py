from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_pdfs(directory: str):
    """
    Load every PDF file from a directory.

    Returns:
        List of LangChain Document objects,
        usually one Document per PDF page.
    """

    documents = []

    pdf_files = sorted(Path(directory).glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in directory: {directory}"
        )

    print(f"Found {len(pdf_files)} PDF files.")

    for pdf_file in pdf_files:
        print(f"Loading: {pdf_file.name}")

        loader = PyPDFLoader(str(pdf_file))

        pages = loader.load()

        for page in pages:
            page.metadata["source"] = pdf_file.name

        documents.extend(pages)

    return documents

def remove_empty_documents(documents):
    """
    Remove pages/documents that contain no useful text.
    """

    cleaned_documents = []

    for document in documents:
        text = document.page_content.strip()

        if text:
            cleaned_documents.append(document)

    return cleaned_documents

def chunk_documents(
    documents,
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
):
    """
    Split loaded PDF pages into smaller chunks.

    Returns:
        List of smaller LangChain Document objects.
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = text_splitter.split_documents(documents)

    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"chunk-{index:05d}"

        page = chunk.metadata.get("page")

        if page is not None:
            chunk.metadata["page_number"] = page + 1

    return chunks