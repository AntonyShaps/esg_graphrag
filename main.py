import streamlit as st
from rag_pipeline import (
    route_to_graphs,
    retrieve_multi,
    answer_with_docs
)


st.set_page_config(page_title="ESG GraphRAG Assistant", layout="wide")

st.title("🔍 ESG GraphRAG Assistant")
st.caption("Ask ESG questions. The router selects relevant company graphs (Meta, Google, Nvidia).")

question = st.text_input(
    "Ask a question:",
    placeholder="e.g. compare google and meta scope 1 emissions"
)

if st.button("Run Query"):
    if not question.strip():
        st.warning("Please enter a question.")
        st.stop()

    # Step 1 – Routing
    st.write("### Step 1 — Routing to graphs...")
    with st.spinner("Selecting relevant graph databases..."):
        companies = route_to_graphs(question)

    st.success(f"Router selected: {companies}")

    st.write("---")
    st.write("### Step 2 — Retrieving relevant chunks from Neo4j")

    # Use your retrieve_multi helper (parallel inside)
    with st.spinner("Retrieving from Neo4j..."):
        results = retrieve_multi(question, companies, k=4)

    st.success("Retrieved data from the selected graphs!")

    # UI: display chunks
    for company, docs in results.items():
        st.write(f"#### Retrieved Documents: {company.upper()}")
        if not docs:
            st.warning(f"No relevant documents found in {company.upper()}.")
            continue

        for d in docs:
            with st.expander(f"Chunk {d['index']} (score={d['score']:.4f})"):
                st.write(d["text"])

    st.write("---")
    st.write("### Step 3 — Generating final answer")

    with st.spinner("Using Mistral (Ollama) to write final answer..."):
        final_answer = answer_with_docs(question, results, model="mistral:7b")

    st.success("Final answer generated!")
    st.write("## Answer:")
    st.write(final_answer)

