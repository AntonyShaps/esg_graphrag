import streamlit as st
from concurrent.futures import ThreadPoolExecutor
import json
from openai import OpenAI
from neo4j import GraphDatabase

# Import your existing functions
from rag_pipeline import (
    embed_model,
    route_to_graphs,
    retrieve_from_company,
    answer_with_docs,
)

# Streamlit page config
st.set_page_config(page_title="ESG GraphRAG Assistant", layout="wide")

# Ollama client
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)

# Neo4j driver
driver = GraphDatabase.driver(
    "neo4j://localhost:7687",
    auth=("neo4j", "graphgraph")
)

st.title("🔍 ESG GraphRAG Assistant")
st.caption("Ask ESG questions. The router selects relevant company graphs (Meta, Google, Nvidia).")

question = st.text_input("Ask a question:", placeholder="e.g. compare google and meta scope 1 emissions")

if st.button("Run Query"):
    if not question.strip():
        st.warning("Please enter a question.")
        st.stop()

    st.write("### Step 1 — Routing to graphs...")
    with st.spinner("Selecting relevant graph databases..."):
        companies = route_to_graphs(question)

    st.success(f"Router selected: {companies}")

    st.write("---")
    st.write("### Step 2 — Retrieving relevant chunks from Neo4j")

    # Parallel retrieval
    # Parallel retrieval for speed
    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(retrieve_from_company, question, comp): comp
            for comp in companies
        }

        results = {
            comp: future.result()
            for future, comp in futures.items()
        }


    st.success("Retrieved data from the selected graphs!")

    # UI: display chunks
    for company in results:
        st.write(f"#### 📚 Retrieved Documents: {company.upper()}")
        docs = results[company]

        if not docs:
            st.warning(f"No relevant documents found in {company.upper()}.")
            continue

        for d in docs:
            with st.expander(f"Chunk {d['index']} (score={d['score']:.4f})"):
                st.write(d["text"])

    st.write("---")
    st.write("### Step 3 — Generating final answer")

    with st.spinner("Using Mistral (Ollama) to write final answer..."):
        final_answer = answer_with_docs(question, results, model="mistral:latest")

    st.success("Final answer generated!")
    st.write("## 🧠 Answer:")
    st.write(final_answer)
