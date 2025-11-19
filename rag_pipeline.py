import json
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer


# ============================================================
# 1. Clients (Ollama + Neo4j)
# ============================================================

# Ollama (OpenAI-compatible)
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)

# Neo4j driver
driver = GraphDatabase.driver(
    "neo4j://localhost:7687",
    auth=("neo4j", "graphgraph")
)

# Embedding model
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# ============================================================
# 2. Router (function-calling via Mistral)
# ============================================================

ROUTER_SYSTEM = """
You are a routing model.
Given a user question, decide which company knowledge graphs must be queried.

Valid companies:
- meta
- google
- nvidia

Return ALL relevant companies. 
If the question mentions multiple companies, return them all.
If uncertain, return the most likely one.
"""

router_tools = [
    {
        "type": "function",
        "function": {
            "name": "select_graphs",
            "description": "Select one or multiple graph DBs needed to answer the user question.",
            "parameters": {
                "type": "object",
                "properties": {
                    "graphs": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["meta", "google", "nvidia"]},
                        "description": "List of graphs to query."
                    }
                },
                "required": ["graphs"]
            }
        }
    }
]


def route_to_graphs(question: str, model="mistral:latest") -> List[str]:
    """
    Router that returns one or multiple graph DBs.
    Example: ["meta"], or ["google", "meta"]
    """
    messages = [
        {"role": "system", "content": ROUTER_SYSTEM},
        {"role": "user", "content": f"Question: {question}"}
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=router_tools,
        tool_choice="auto",
        temperature=0,
    )

    msg = response.choices[0].message

    if not msg.tool_calls:
        return ["meta"]  # fallback

    args = json.loads(msg.tool_calls[0].function.arguments)

    # Guarantee type
    graphs = args.get("graphs", [])
    if not graphs:
        return ["meta"]

    return graphs


# ============================================================
# 3. Neo4j Retrieval Functions
# ============================================================

def retrieve_from_company(question: str, company: str, k: int = 4) -> List[Dict]:
    """
    Retrieve chunks from the selected company's Neo4j DB.
    Assumes each DB has a vector index named the same as the DB: meta, google, nvidia.
    """
    embedding = embed_model.encode([question])[0].tolist()

    cypher = f"""
    CALL db.index.vector.queryNodes('{company}', $k, $embedding)
    YIELD node AS hits, score
    RETURN hits.text AS text, score, hits.index AS index
    """

    with driver.session(database=company) as session:
        result = session.run(cypher, embedding=embedding, k=k)
        records = list(result)

    return [
        {
            "text": r["text"],
            "score": r["score"],
            "index": r["index"],
            "company": company,
        }
        for r in records
    ]


# ============================================================
# 4. Answer Generation
# ============================================================

ANSWER_SYSTEM = """
You are an ESG expert.
You must answer ONLY using the provided documents.
If the answer is not found in the documents, say:
"I don't know based on these documents."
"""

def answer_with_docs(question: str, docs_by_company: dict, model="mistral:latest") -> str:
    """
    docs_by_company = { "google": [...], "meta": [...], "nvidia": [...] }
    """
    all_docs = []
    for company, docs in docs_by_company.items():
        for d in docs:
            all_docs.append(f"[{company.upper()}] {d['text']}")

    user_message = f"""
Use ONLY the following documents:

{json.dumps(all_docs, indent=2)}

---

Question: {question}
"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": ANSWER_SYSTEM},
            {"role": "user", "content": user_message},
        ],
        temperature=0,
    )

    return response.choices[0].message.content


# ============================================================
# 5. Main RAG Orchestrator
# ============================================================

def run_rag(question: str) -> str:
    """
    1. Route → choose companies
    2. Retrieve each company's chunks (parallel)
    3. Answer using Mistral
    """
    companies = route_to_graphs(question)

    # Parallel retrieval for speed
    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(retrieve_from_company, question, comp): comp
            for comp in companies
        }
        docs = {
            futures[f]: futures[f].result()
            for f in futures
        }

    # Convert "future: company" → "company: chunks"
    final_docs = {}
    for f, comp in futures.items():
        final_docs[comp] = docs[f]

    # LLM final answer
    final_answer = answer_with_docs(question, final_docs)
    return final_answer
