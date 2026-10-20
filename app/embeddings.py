"""Embedding generation using Google's Gemini embedding model."""

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.config import get_settings


def get_embedding_model(api_key: str | None = None) -> GoogleGenerativeAIEmbeddings:
    """Return an initialized GoogleGenerativeAIEmbeddings instance.

    Raises:
        ValueError: If no Gemini API key is provided or configured.
    """
    settings = get_settings()
    key = api_key if api_key is not None else settings.gemini_api_key
    if not key:
        raise ValueError("GEMINI_API_KEY is not configured. Please set it in your .env file.")

    model_name = settings.embedding_model
    # Ensure models/ prefix if not present for Google GenAI compatibility
    if not model_name.startswith("models/"):
        model_name = f"models/{model_name}"

    return GoogleGenerativeAIEmbeddings(
        model=model_name,
        google_api_key=key,
        output_dimensionality=settings.embedding_dimensions,
    )


def embed_texts(
    texts: list[str],
    model: GoogleGenerativeAIEmbeddings | None = None,
) -> list[list[float]]:
    """Generate embedding vectors for a list of text strings.

    Returns an empty list if given an empty list of texts.
    """
    if not texts:
        return []

    settings = get_settings()
    embedding_model = model or get_embedding_model()
    return embedding_model.embed_documents(
        texts, output_dimensionality=settings.embedding_dimensions
    )


def embed_query(
    query: str,
    model: GoogleGenerativeAIEmbeddings | None = None,
) -> list[float]:
    """Generate an embedding vector for a single query string."""
    clean_query = query.strip()
    if not clean_query:
        raise ValueError("Query string cannot be empty.")

    settings = get_settings()
    embedding_model = model or get_embedding_model()
    return embedding_model.embed_query(
        clean_query, output_dimensionality=settings.embedding_dimensions
    )
