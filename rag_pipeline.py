import json
from typing import List, Dict

import concurrent.futures
from openai import OpenAI
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer



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



# helper function 
def chat(messages, model, temperature=0, **config):
    """Generic chat call, returns full message object."""
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=messages,
        **config,
    )
    return response.choices[0].message

router_tools = [
    {
        "type": "function",
        "function": {
            "name": "select_graphs",
            "description": "Return which company graphs to query based on the question.",
            "parameters": {
                "type": "object",
                "properties": {
                    "graphs": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["meta", "google", "nvidia"]
                        },
                        "description": "List of relevant company graphs."
                    }
                },
                "required": ["graphs"]
            }
        }
    }
]


def route_to_graphs(question: str) -> list[str]:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a router. Decide which company graphs are relevant "
                "to answer the question. Companies: meta, google, nvidia. "
                "Return all that apply."
            )
        },
        {"role": "user", "content": f"Question: {question}"}
    ]

    resp = client.chat.completions.create(
        model="mistral:latest",
        messages=messages,
        tools=router_tools,
        tool_choice="auto",
        temperature=0,
    )

    msg = resp.choices[0].message

    args = json.loads(msg.tool_calls[0].function.arguments)
    return args["graphs"]


def retrieve_from_company(question: str, company_name: str, k: int = 4) -> list[dict]:
    embedding = embed_model.encode([question])[0].tolist()

    cypher = f"""
    CALL db.index.vector.queryNodes('{company_name}', $k, $embedding)
    YIELD node AS hits, score
    RETURN hits.text AS text, score, hits.index AS index
    """

    with driver.session(database=company_name) as session:
        result = session.run(cypher, embedding=embedding, k=k)
        records = list(result)

    return [
        {
            "text": r["text"],
            "score": r["score"],
            "index": r["index"]
        }
        for r in records
    ]




def retrieve_multi(question: str, companies: list[str], k: int = 4):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(retrieve_from_company, question, company): company
            for company in companies
        }
        results = {}
        for future in futures:
            company = futures[future]
            results[company] = future.result()
        return results
    

ANSWER_SYSTEM_PROMPT = """
You are an ESG analyst.
Use ONLY the provided documents.
If something is missing, say: “I don’t know based on these documents.”
"""

def answer_with_docs(question: str, retrieved: dict, model="mistral:latest"):
    text = ""

    for company, docs in retrieved.items():
        text += f"\n### {company.upper()} DOCUMENTS:\n"
        for d in docs:
            text += d["text"] + "\n"

    final_prompt = f"""
Use the documents below to answer the question.
If not enough info, say you don't know.

Documents:
{text}

Question: {question}
"""

    msg = chat(
        [
            {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
            {"role": "user", "content": final_prompt},
        ],
        model=model,
        temperature=0
    )
    return msg.content
