import os
import re
from pypdf import PdfReader

PDF_FOLDER = os.path.join("data", "pdfs")

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


def is_reference_page(text):
    """
    Detect pages that appear to belong to a references section.
    """

    text_lower = text.lower().strip()

    first_part = text_lower[:500]

    reference_headings = [
        "references",
        "bibliography",
        "reference list"
    ]

    # Explicit reference heading
    for heading in reference_headings:
        if heading in first_part:
            return True

    # Count strong bibliography signals across the whole page
    signals = 0

    patterns = [
        r"\[\d+\]",              # [1], [23]
        r"\[crossref\]",         # [CrossRef]
        r"arxiv:\d+",            # arXiv:1234.5678
        r"doi\.org",
        r"\bdoi:",
        r"\bet al\.",
    ]

    for pattern in patterns:
        signals += len(re.findall(pattern, text_lower))

    # A page with several bibliography patterns is almost certainly references
    if signals >= 5:
        return True

    return False


def is_reference_chunk(text):
    """
    Detect chunks that are primarily bibliography/reference content.
    """

    text_lower = text.lower()

    score = 0

    # Numbered citation entries
    numbered_citations = len(
        re.findall(r"\[\d+\]", text)
    )

    if numbered_citations >= 2:
        score += 2

    # CrossRef entries
    crossref_count = len(
        re.findall(r"\[crossref\]", text_lower)
    )

    if crossref_count >= 1:
        score += 3

    # arXiv references
    arxiv_count = len(
        re.findall(r"arxiv", text_lower)
    )

    if arxiv_count >= 2:
        score += 2

    # DOI references
    doi_count = len(
        re.findall(r"doi", text_lower)
    )

    if doi_count >= 1:
        score += 2

    # Bibliography-style phrases
    bibliography_terms = [
        "proceedings",
        "conference",
        "journal",
        "vol.",
        "preprint",
        "early access",
        "crossref"
    ]

    for term in bibliography_terms:
        if term in text_lower:
            score += 1

    # If enough bibliography evidence exists, reject the chunk
    return score >= 4


def load_pdfs():
    """
    Load all PDFs and extract text page by page.
    """

    documents = []
    total_pages = 0
    skipped_pages = 0

    print("=" * 60)
    print("MULTI-DOCUMENT RAG - PDF INGESTION")
    print("=" * 60)

    if not os.path.exists(PDF_FOLDER):
        print(f"\nERROR: PDF folder not found: {PDF_FOLDER}")
        return documents

    pdf_files = [
        file
        for file in os.listdir(PDF_FOLDER)
        if file.lower().endswith(".pdf")
    ]

    if not pdf_files:
        print("\nNo PDF files found!")
        return documents

    print(f"\nPDFs found: {len(pdf_files)}\n")

    for pdf_file in pdf_files:

        pdf_path = os.path.join(PDF_FOLDER, pdf_file)

        try:
            reader = PdfReader(pdf_path)
            num_pages = len(reader.pages)

            print("-" * 60)
            print(f"Loading: {pdf_file}")
            print(f"Pages: {num_pages}")

            total_pages += num_pages
            extracted_pages = 0

            for page_number, page in enumerate(
                reader.pages,
                start=1
            ):

                text = page.extract_text()

                if not text or not text.strip():
                    continue

                cleaned_text = text.strip()

                # Skip reference-heavy pages
                if is_reference_page(cleaned_text):

                    print(
                        f"Skipping reference page: "
                        f"{pdf_file} - Page {page_number}"
                    )

                    skipped_pages += 1
                    continue

                documents.append({
                    "text": cleaned_text,
                    "document": pdf_file,
                    "page": page_number
                })

                extracted_pages += 1

            print(
                f"Successfully extracted text from "
                f"{extracted_pages} usable pages."
            )

        except Exception as e:
            print(f"\nERROR loading: {pdf_file}")
            print(f"Error: {e}")

    print("\n" + "=" * 60)
    print("INGESTION COMPLETE")
    print("=" * 60)

    print(f"Total PDFs processed: {len(pdf_files)}")
    print(f"Total pages processed: {total_pages}")
    print(f"Pages with usable text: {len(documents)}")
    print(f"Reference pages skipped: {skipped_pages}")

    return documents


def chunk_documents(documents):
    """
    Split documents into sentence-aware overlapping chunks.
    """

    chunks = []
    skipped_chunks = 0

    print("\n" + "=" * 60)
    print("SENTENCE-AWARE CHUNKING DOCUMENTS")
    print("=" * 60)

    for doc in documents:

        text = doc["text"]

        # Clean unnecessary line breaks
        text = re.sub(r"\s+", " ", text)

        # Split text into sentences
        sentences = re.split(
            r"(?<=[.!?])\s+",
            text
        )

        current_chunk = ""
        chunk_number = 1

        for sentence in sentences:

            sentence = sentence.strip()

            if not sentence:
                continue

            # If adding the sentence exceeds chunk size,
            # save the current chunk first
            if (
                len(current_chunk) + len(sentence)
                > CHUNK_SIZE
                and current_chunk
            ):

                document_name = os.path.splitext(
                    doc["document"]
                )[0]

                chunk_id = (
                    f"{document_name}"
                    f"_page{doc['page']}"
                    f"_chunk{chunk_number}"
                )

                # Check whether chunk is bibliography/reference content
                if is_reference_chunk(current_chunk):

                    print(
                        f"Skipping reference chunk: "
                        f"{chunk_id}"
                    )

                    skipped_chunks += 1

                else:

                    chunks.append({
                        "chunk_id": chunk_id,
                        "text": current_chunk.strip(),
                        "document": doc["document"],
                        "page": doc["page"]
                    })

                chunk_number += 1

                # Create overlap from the end of previous chunk
                overlap_text = current_chunk[
                    -CHUNK_OVERLAP:
                ]

                current_chunk = overlap_text + " " + sentence

            else:

                current_chunk += " " + sentence

        # Save the final chunk from the page
        if len(current_chunk.strip()) >= 100:

            document_name = os.path.splitext(
                doc["document"]
            )[0]

            chunk_id = (
                f"{document_name}"
                f"_page{doc['page']}"
                f"_chunk{chunk_number}"
            )

            if is_reference_chunk(current_chunk):

                print(
                    f"Skipping reference chunk: "
                    f"{chunk_id}"
                )

                skipped_chunks += 1

            else:

                chunks.append({
                    "chunk_id": chunk_id,
                    "text": current_chunk.strip(),
                    "document": doc["document"],
                    "page": doc["page"]
                })

    print("\n" + "-" * 60)
    print(f"Total usable chunks created: {len(chunks)}")
    print(f"Reference chunks skipped: {skipped_chunks}")

    return chunks


if __name__ == "__main__":

    documents = load_pdfs()

    chunks = chunk_documents(documents)

    if chunks:

        print("\n" + "=" * 60)
        print("CHUNK PREVIEW")
        print("=" * 60)

        preview = chunks[0]

        print(f"\nChunk ID: {preview['chunk_id']}")
        print(f"Document: {preview['document']}")
        print(f"Page: {preview['page']}")

        print("\nChunk Text:\n")
        print(preview["text"])

    else:
        print("\nNo usable chunks were created.")