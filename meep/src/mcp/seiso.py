
from typing import Literal, List

import os, chromadb, langchain_openai, chromadb.utils.embedding_functions, langchain_openai.embeddings

# chroma_compatible_google_ef = chromadb.utils.embedding_functions.ChromaLangchainEmbeddingFunction(
#     embedding_function = langchain_google_genai.GoogleGenerativeAIEmbeddings(
#         model = "models/text-embedding-004"
#     )
# )

chroma_compatible_openai_ef = chromadb.utils.embedding_functions.ChromaLangchainEmbeddingFunction(
    embedding_function = langchain_openai.embeddings.OpenAIEmbeddings(
        model="text-embedding-3-small",
        chunk_size=1,  # ChromaDB requires chunk_size=1 for compatibility
        api_key=os.environ["OPENAI_API_KEY"]
    )
)

client = chromadb.PersistentClient(path=os.path.join(os.getcwd(), "data", "databases", "chroma_seiso"))

roots_collection = client.get_or_create_collection(name="roots", embedding_function=chroma_compatible_openai_ef)
nouns_collection = client.get_or_create_collection(name="nouns", embedding_function=chroma_compatible_openai_ef)
compounds_collection = client.get_or_create_collection(name="compounds", embedding_function=chroma_compatible_openai_ef)


def recompute_embeddings(batch_size: int = 100):
    for collection in [roots_collection, nouns_collection, compounds_collection]:
        result = ""
        
        count = collection.count()
        if count == 0:
            result += f"Collection '{collection.name}' is empty. Skipping.\n"
        else:
            for offset in range(0, count, batch_size):
                print(f"Processing batch: documents {offset} to {offset + batch_size - 1}...")

                # We fetch the original documents to re-embed them.
                batch_data = collection.get(
                    limit=batch_size,
                    offset=offset,
                    include=["metadatas", "documents"] 
                )

                # By providing documents but NOT embeddings, we force ChromaDB to use
                # the collection's embedding function to generate new vectors.
                collection.upsert(
                    ids=batch_data['ids'],
                    documents=batch_data['documents'],
                    metadatas=batch_data['metadatas']
                )
            
            results += f"Recomputed embeddings for collection '{collection.name}' with {count} documents.\n"
    
    return result if result else "No collections to recompute embeddings for."
            
def upsert_root(root: str, description: str, type: Literal["initial", "final"]):
    """Insert or update a root into the database."""
    roots_collection.upsert(
        documents=[description],
        metadatas=[{
            "root": root.lower().strip(), 
            "description": description, 
            "type": type.lower().strip()
        }],
        ids=[root.lower().strip()]
    )
    return f"Upserted root {root} with description {description} and type {type}."

def upsert_noun(initial: str, final: str, description: str):
    """Insert or update a noun into the database."""
    nouns_collection.upsert(
        documents=[description],
        metadatas=[{
            "initial": initial.lower().strip(), 
            "final": final.lower().strip(), 
            "description": description
        }],
        ids=[f"{initial.lower().strip()}{final.lower().strip()}"]
    )
    return f"Upserted noun {initial}+{final} with description {description}."
    
def upsert_compound(sequence: str, translation: str):
    """Insert or update a compound into the database."""
    compounds_collection.upsert(
        documents=[translation],
        metadatas=[{
            "sequence": sequence.lower().strip(), 
            "translation": translation.lower().strip()
        }],
        ids=[sequence.lower().strip()]
    )
    return f"Upserted compound {sequence} with translation {translation}."

def delete_root(root: str):
    """Delete a root from the database."""
    roots_collection.delete(ids=[root.lower().strip()])
    return f"Deleted root {root}."

def delete_noun(initial: str, final: str):
    """Delete a noun from the database."""
    nouns_collection.delete(ids=[f"{initial.lower().strip()}{final.lower().strip()}"])
    return f"Deleted noun {initial}+{final}."

def delete_compound(sequence: str):
    """Delete a compound from the database."""
    compounds_collection.delete(ids=[sequence.lower().strip()])
    return f"Deleted compound {sequence}."

def search_from_natural_language(q: str) -> str:
    """Search the seiso dictionary for roots, nouns, or compounds in the database using natural language. (= from translations)"""
    
    output_string = ""
    
    for results, formatted in zip(
        [
            roots_collection.query(query_texts=[q], n_results=5),
            nouns_collection.query(query_texts=[q], n_results=5),
            compounds_collection.query(query_texts=[q], n_results=5)
        ],
        [
            lambda r: ''.join([f'{mt["root"]} - {mt["description"]} ({mt["type"]})\n' for mt in r["metadatas"][0]]),
            lambda r: ''.join([f'{mt["initial"]}+{mt["final"]} - {mt["description"]}\n' for mt in r["metadatas"][0]]),
            lambda r: ''.join([f'{mt["sequence"]} - {mt["translation"]}\n' for mt in r["metadatas"][0]])
        ]
    ):
        output_string += "\n" + formatted(results) if "metadatas" in results.keys() else ""
    
    return output_string

def search_from_seiso(q: str) -> str:
    """Search the seiso dictionary for roots, nouns, or compounds from their seiso form. (= from direct seiso)"""
    
    output_string = ""
    
    # Search the 3 collections' by id in a "%like%" fashion using python's "in" operator on all items
    for results, formatted in zip(
        [
            [mt for mt in roots_collection.get(include=["metadatas"])["metadatas"] if q.lower().strip() in mt["root"].lower().strip()],
            [mt for mt in nouns_collection.get(include=["metadatas"])["metadatas"] if q.lower().strip() in f"{mt['initial'].lower().strip()}{mt['final'].lower().strip()}"],
            [mt for mt in compounds_collection.get(include=["metadatas"])["metadatas"] if q.lower().strip() in mt["sequence"].lower().strip()]
        ],
        [
            lambda r: ''.join([f'{mt["root"]} - {mt["description"]} ({mt["type"]})\n' for mt in r]),
            lambda r: ''.join([f'{mt["initial"]}+{mt["final"]} - {mt["description"]}\n' for mt in r]),
            lambda r: ''.join([f'{mt["sequence"]} - {mt["translation"]}\n' for mt in r])
        ]
    ):
        output_string += "\n" + formatted(results)
    
    return output_string

tools: List[callable] = [
    upsert_root,
    upsert_noun,
    upsert_compound,
    delete_root,
    delete_noun,
    delete_compound,
    search_from_natural_language,
    search_from_seiso,
    recompute_embeddings
]
