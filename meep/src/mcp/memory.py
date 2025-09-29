
from typing import List

import pydantic, data.jsondb

class Document(pydantic.BaseModel):
    description: str
    contraintes: str = ""
    content: str = ""
    def __hash__(self): return hash(self.description)
    def __len__(self): return len(self.content)
    def __str__(self): return self.description + "\n" + self.content + " (" + str(len(self)) + " words)"

class Database(pydantic.BaseModel):
    documents: List[Document] = []

DB = data.jsondb.JsonDB(
    path="./data/long_term_memory/memory.json",
    model=Database
)

async def afficher_plus_de_documents(id_afficher: List[int]):
    """Permet de demander l'affichage du contenu d'un espace de la mémoire."""
    async with DB as db:
        return "Meep requested to view the content of the following memory spaces: " "\n".join(
            f"\n\ndescription={db.documents[i].description}:\n"
            for i in id_afficher if 0 <= i < len(db.documents)
        )

async def creer_espace_memoire(description: str, contraintes: str):
    """Crée un espace mémoire avec une description qui indique ce que l'on peut y stocker, et des contraintes qui indiquent comment filtrer et traiter l'information qui y entre."""
    async with DB as db:
        db.documents.append(
            Document(
                description=description,
                contraintes=contraintes,
                chunks=[]
            )
        )
    return f"Espace mémoire créé avec la description: {description} et les contraintes: {contraintes}"

async def supprimer_espace_memoire(id_espace: int):
    """Permet de supprimer un espace mémoire."""
    async with DB as db:
        if 0 <= id_espace < len(db.documents):
            removed_doc = db.documents.pop(id_espace)
            return f"Espace mémoire supprimé: {removed_doc.description}"
        else:
            return f"Aucun espace mémoire trouvé avec l'ID {id_espace}."

tools: List[callable] = [
    afficher_plus_de_documents,
    creer_espace_memoire,
    supprimer_espace_memoire
]
