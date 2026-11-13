
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    NLTKTextSplitter,
    MarkdownHeaderTextSplitter
)

# Sample document
document = """
Python is a programming language.
Python supports object-oriented programming.

Machine learning is a branch of artificial intelligence.
Deep learning is a subset of machine learning.
"""


# ------------------------------------------------------------
# 1. FIXED-SIZE CHUNKING
# ------------------------------------------------------------

# CharacterTextSplitter divides text into chunks using
# a specified separator and character-based size.

fixed_splitter = CharacterTextSplitter(
    separator=" ",
    chunk_size=100,
    chunk_overlap=0
)

fixed_chunks = fixed_splitter.split_text(document)

print("\n1. FIXED-SIZE CHUNKING")

for chunk in fixed_chunks:
    print(chunk)


# ------------------------------------------------------------
# 2. CHUNKING WITH OVERLAP
# ------------------------------------------------------------

# Overlap preserves some text between consecutive chunks.
# This helps maintain continuity.

overlap_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)

overlap_chunks = overlap_splitter.split_text(document)

print("\n2. CHUNKING WITH OVERLAP")

for chunk in overlap_chunks:
    print(chunk)


# ------------------------------------------------------------
# 3. SENTENCE-BASED CHUNKING
# ------------------------------------------------------------

# NLTKTextSplitter uses sentence tokenization.
# It groups sentences while respecting the configured
# chunk size and overlap.

sentence_splitter = NLTKTextSplitter(
    chunk_size=100,
    chunk_overlap=10
)

sentence_chunks = sentence_splitter.split_text(document)

print("\n3. SENTENCE-BASED CHUNKING")

for chunk in sentence_chunks:
    print(chunk)


# ------------------------------------------------------------
# 4. RECURSIVE CHUNKING
# ------------------------------------------------------------

# RecursiveCharacterTextSplitter tries different separators
# to preserve paragraphs and sentences where possible.

recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    separators=["\n\n", "\n", ". ", " ", ""]
)

recursive_chunks = recursive_splitter.split_text(document)

print("\n4. RECURSIVE CHUNKING")

for chunk in recursive_chunks:
    print(chunk)


# ------------------------------------------------------------
# 5. SEMANTIC CHUNKING
# ------------------------------------------------------------

# SemanticChunker groups text based on changes in meaning.
# It uses embeddings to calculate semantic similarity.

from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

semantic_splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="percentile"
)

semantic_chunks = semantic_splitter.split_text(document)

print("\n5. SEMANTIC CHUNKING")

for chunk in semantic_chunks:
    print(chunk)


# ------------------------------------------------------------
# 6. STRUCTURE-AWARE CHUNKING
# ------------------------------------------------------------

# MarkdownHeaderTextSplitter divides Markdown according
# to its heading hierarchy and preserves heading metadata.

markdown_document = """
# Python

Python is a programming language.

## Python Applications

Python is used in AI and data science.

# Machine Learning

Machine learning learns patterns from data.
"""

headers_to_split_on = [
    ("#", "Section"),
    ("##", "Subsection")
]

structure_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on
)

structured_chunks = structure_splitter.split_text(
    markdown_document
)

print("\n6. STRUCTURE-AWARE CHUNKING")

for chunk in structured_chunks:
    print("Content:", chunk.page_content)
    print("Metadata:", chunk.metadata)
    print("-----------------------")
