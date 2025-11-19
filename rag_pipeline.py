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
    "bolt://localhost:7687",
    auth=("neo4j", "graphgraph")
)

# Embedding model
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")



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
ROUTER_SYSTEM_PROMPT = """
You MUST call the tool `select_graphs`.

Your ONLY task is:
- Read the question
- Decide which company names appear: ["meta", "google", "nvidia"]
- Return them in the parameter `graphs` as a list of strings.

RULES:
- ALWAYS call the tool. NEVER answer directly.
- If the question contains multiple companies, include all of them.
- If the question does not mention any company, return ["meta"].
- Do NOT think, do NOT explain, do NOT justify.
- You MUST output a tool call.
"""



def route_to_graphs(question: str) -> list[str]:
    messages = [
        {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    resp = client.chat.completions.create(
        model="mistral:7b",
        messages=messages,
        tools=router_tools,
        tool_choice="auto",   # forces tool
        temperature=0,
    )

    msg = resp.choices[0].message

    # SAFE extraction
    if msg.tool_calls:
        args = json.loads(msg.tool_calls[0].function.arguments)
        return args["graphs"]

    # ABSOLUTE fallback (should never happen with strict prompt)
    lower = question.lower()
    found = [c for c in ["meta", "google", "nvidia"] if c in lower]
    return found if found else ["meta"]



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
You are an ESG analyst answering questions ONLY from the provided retrieved documents.

The documents come from OCR extraction and follow these rules:
- Tables are represented in HTML format (e.g., <table>...</table>). Interpret and use them as actual tables.
- Equations appear in LaTeX format. Render or reference them normally.
- Images appear inside <img></img> tags. If these contain a short description, treat it as an image caption.
- Page numbers are marked as <page_number></page_number>.
- Footers appear as <footer></footer>.
- Checkboxes use ☐ (empty) and ☑ (checked).

Your task:
1. Use ONLY the information contained inside the provided document chunks.
2. Correctly interpret HTML tables and extract meaningful data from them.
3. When giving values from tables, present them in clean Markdown format.
4. If the required information is not explicitly stated in the documents, respond with:
   "I don’t know based on these documents."
5. DO NOT hallucinate, infer, or invent numbers or content not shown in the documents.
6. If multiple companies' documents are provided, distinguish them clearly in your reasoning and your final answer.
7. Keep the answer concise, factual, and directly tied to the retrieved data.
"""


def answer_with_docs(question: str, retrieved: dict, model="mistral:7b"):
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
