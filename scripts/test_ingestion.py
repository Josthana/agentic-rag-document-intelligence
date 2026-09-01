from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_pdfs(directory: str):
    """
    Load all PDF files from the given directory.

    Returns a list of LangChain Document objects.
    """

    documents = []

    pdf_files = sorted(
        Path(directory).glob("*.pdf")
    )

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in: {directory}"
        )

    print(
        f"Found {len(pdf_files)} PDF file(s)."
    )

    for pdf_file in pdf_files:

        print(
            f"Loading: {pdf_file.name}"
        )

        loader = PyPDFLoader(
            str(pdf_file)
        )

        pages = loader.load()

        for page in pages:

            page.metadata["source"] = (
                pdf_file.name
            )

            raw_page_number = (
                page.metadata.get("page")
            )

            if raw_page_number is not None:
                page.metadata["page_number"] = (
                    raw_page_number + 1
                )

        documents.extend(pages)

    return documents


def remove_empty_documents(documents):
    """
    Remove documents/pages that contain no useful text.
    """

    cleaned_documents = []

    for document in documents:

        text = document.page_content.strip()

        if text:
            cleaned_documents.append(
                document
            )

    return cleaned_documents


def chunk_documents(
    documents,
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
):
    """
    Split documents into smaller chunks.
    """

    text_splitter = (
        RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                "",
            ],
        )
    )

    chunks = (
        text_splitter.split_documents(
            documents
        )
    )

    for index, chunk in enumerate(chunks):

        chunk.metadata["chunk_id"] = (
            f"chunk-{index:05d}"
        )

    return chunks