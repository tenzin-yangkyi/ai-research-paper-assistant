# AI Research Paper Assistant

A Retrieval-Augmented Generation (RAG) application that allows users to upload academic research papers and interact with them using Google's Gemini API.

## Features

- Upload and process multiple research paper PDFs.
- Ask questions using semantic search.
- Compare research methodologies and findings.
- Generate literature review summaries.
- Explore potential research gaps.
- Retrieve supporting passages with paper names and page numbers.
- Stream Gemini responses to improve perceived latency.

## Tech Stack

- Python
- Streamlit
- Google Gemini API
- Sentence Transformers
- ChromaDB
- PyPDF

## Setup

1. Clone this repository.
2. Create and activate a Python virtual environment.
3. Install dependencies using `pip install -r requirements.txt`.
4. Set your `GEMINI_API_KEY` environment variable.
5. Run `streamlit run app.py`.

## Project Status

Prototype. Retrieval quality, citation accuracy, and performance should be evaluated before relying on generated research summaries.

## Security

Never commit API keys, confidential documents, or private research data to the repository.