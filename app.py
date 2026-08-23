"""PowerOps — AI-powered DevOps Management Assistant (Streamlit POC).

Wires the retrieval + grounding + escalation pipeline built in Notebooks
01-06 into an interactive UI. This file intentionally keeps all pipeline
logic inline (rather than importing from a src/ package) since the src/
refactor is a separate, later step — see notebooks/06_human_escalation.ipynb
for the notebook this logic was validated in.
"""

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
VOCAB_PATH = DATA_DIR / "vocabulary.json"
ESCALATIONS_PATH = DATA_DIR / "escalations.json"

load_dotenv(dotenv_path=BASE_DIR / ".env")

TOP_K = int(os.environ.get("TOP_K", "5"))

EXAMPLE_QUESTIONS = [
    "Which high-priority problems belong to Nova Team?",
    "What is happening with INO-21920?",
    "What Rejected issues does Summit Crew have?",
    "What issues does Gibson, Anjali have?",
    "What issues are currently On Hold?",
    "What critical issues are currently open?",
    "What issues does John currently own?",
    "What issues does Anjali have?",
]

ISSUE_KEY_PATTERN = re.compile(r"\bINO-\d+\b", re.IGNORECASE)

SYSTEM_PROMPT = """You are PowerOps, a DevOps management assistant.

Answer the user's question using ONLY the supplied PowerOps context below.

Do not invent issues, statuses, assignees, priorities, teams, dates, or \
resolutions that are not explicitly present in the context.

When referencing an issue, include its Issue Key (format: INO-#####).

If the available context does not contain enough evidence to answer the \
question, explicitly state that sufficient information was not found in \
the PowerOps knowledge base. Do not guess or fill gaps.

For questions requesting multiple issues, provide a concise table when \
appropriate (columns: Issue Key, Summary, Team, Assignee, Priority, Status)."""

LLM_DECLINE_PHRASES = [
    "sufficient information was not found",
    "not enough information",
    "insufficient information",
    "could not find",
    "cannot determine",
    "can't determine",
    "no information",
    "not found in the powerops knowledge base",
    "unable to determine",
]


# ---------------------------------------------------------------------------
# Cached resources (built once per session, not on every rerun)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_vocabulary() -> dict:
    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource(show_spinner="Connecting to Pinecone and OpenAI...")
def get_clients():
    missing = [k for k in ("PINECONE_API_KEY", "OPENAI_API_KEY") if not os.environ.get(k)]
    if missing:
        raise RuntimeError(f"Missing required environment variable(s): {', '.join(missing)}")

    pinecone_index_name = os.environ.get("PINECONE_INDEX_NAME", "powerops-v1")
    embedding_model = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
    chat_model = os.environ.get("CHAT_MODEL", "gpt-4o-mini")
    temperature = float(os.environ.get("LLM_TEMPERATURE", "0.0"))

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index(pinecone_index_name)
    embeddings = OpenAIEmbeddings(model=embedding_model, api_key=os.environ["OPENAI_API_KEY"])
    vector_store = PineconeVectorStore(index=index, embedding=embeddings)
    llm = ChatOpenAI(model=chat_model, temperature=temperature)
    return vector_store, llm


# ---------------------------------------------------------------------------
# Deterministic query parsing (identical logic to Notebook 04/05/06)
# ---------------------------------------------------------------------------

def find_issue_key(question: str) -> str | None:
    match = ISSUE_KEY_PATTERN.search(question)
    return match.group(0).upper() if match else None


def find_vocab_match(question: str, values: list[str]) -> str | None:
    q_lower = question.lower()
    for value in sorted(values, key=len, reverse=True):
        pattern = r"\b" + re.escape(value.lower()) + r"\b"
        if re.search(pattern, q_lower):
            return value
    return None


def split_name(assignee: str) -> tuple[str, str]:
    cleaned = assignee.replace("(Contractor)", "").strip()
    if "," in cleaned:
        last, first = (p.strip() for p in cleaned.split(",", 1))
    else:
        parts = cleaned.split()
        last, first = (parts[0], " ".join(parts[1:])) if parts else ("", "")
    return last, first


def find_assignee(question: str, name_parts: dict) -> dict:
    q_lower = question.lower()

    full_matches = set()
    for assignee, (last, first) in name_parts.items():
        if not last or not first:
            continue
        if (re.search(r"\b" + re.escape(last.lower()) + r"\b", q_lower)
                and re.search(r"\b" + re.escape(first.lower()) + r"\b", q_lower)):
            full_matches.add(assignee)
    if len(full_matches) == 1:
        return {"assignee": next(iter(full_matches)), "ambiguous_candidates": None}
    if len(full_matches) > 1:
        return {"assignee": None, "ambiguous_candidates": sorted(full_matches)}

    token_matches = set()
    for assignee, (last, first) in name_parts.items():
        for token in (first, last):
            if token and len(token) >= 3 and re.search(r"\b" + re.escape(token.lower()) + r"\b", q_lower):
                token_matches.add(assignee)
                break
    if len(token_matches) == 1:
        return {"assignee": next(iter(token_matches)), "ambiguous_candidates": None}
    if len(token_matches) > 1:
        return {"assignee": None, "ambiguous_candidates": sorted(token_matches)}
    return {"assignee": None, "ambiguous_candidates": None}


def parse_query_filters(question: str, vocabulary: dict, name_parts: dict) -> dict:
    filters: dict = {}
    matched_spans = []

    issue_key = find_issue_key(question)
    if issue_key:
        filters["issue_key"] = issue_key
        matched_spans.append(issue_key)

    team = find_vocab_match(question, vocabulary["assigned_teams"])
    if team:
        filters["assigned_team"] = team
        matched_spans.append(team)

    priority = find_vocab_match(question, vocabulary["priorities"])
    if priority:
        filters["priority"] = priority
        matched_spans.append(priority)

    status = find_vocab_match(question, vocabulary["statuses"])
    if status:
        filters["status"] = status
        matched_spans.append(status)

    assignee_result = find_assignee(question, name_parts)
    if assignee_result["assignee"]:
        filters["assignee"] = assignee_result["assignee"]
        last, first = name_parts[assignee_result["assignee"]]
        matched_spans.extend([t for t in (last, first) if t])

    semantic_query = question
    for span in matched_spans:
        semantic_query = re.sub(re.escape(span), "", semantic_query, flags=re.IGNORECASE)
    semantic_query = re.sub(r"\s+", " ", semantic_query).strip(" ?.")
    if not semantic_query:
        semantic_query = question

    return {
        "filters": filters,
        "ambiguous_assignee_candidates": assignee_result["ambiguous_candidates"],
        "semantic_query": semantic_query,
    }


def build_pinecone_filter(filters: dict) -> dict:
    pinecone_filter = {}
    for key in ("assigned_team", "assignee", "priority", "status", "issue_key"):
        if key in filters:
            pinecone_filter[key] = {"$eq": filters[key]}
    return pinecone_filter


def retrieve_powerops_documents(question: str, vector_store, vocabulary: dict, name_parts: dict, top_k: int = TOP_K) -> dict:
    parsed = parse_query_filters(question, vocabulary, name_parts)
    pinecone_filter = build_pinecone_filter(parsed["filters"])
    results = vector_store.similarity_search_with_score(
        parsed["semantic_query"], k=top_k, filter=pinecone_filter or None,
    )
    return {
        "question": question,
        "filters": parsed["filters"],
        "ambiguous_assignee_candidates": parsed["ambiguous_assignee_candidates"],
        "semantic_query": parsed["semantic_query"],
        "pinecone_filter": pinecone_filter,
        "results": results,
    }


def build_context(docs: list) -> str:
    blocks = [f"[Result {i + 1}]\n{doc.page_content}" for i, doc in enumerate(docs)]
    return "\n\n---\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Evidence evaluation + escalation (identical logic to Notebook 06)
# ---------------------------------------------------------------------------

def llm_declined(answer: str) -> bool:
    answer_lower = answer.lower()
    return any(phrase in answer_lower for phrase in LLM_DECLINE_PHRASES)


def unsupported_citations(answer: str, retrieved_issue_keys: set) -> list[str]:
    cited = {m.upper() for m in ISSUE_KEY_PATTERN.findall(answer)}
    return sorted(cited - retrieved_issue_keys)


def evaluate_evidence(retrieved: list, filters: dict, ambiguous_assignee_candidates, answer: str) -> dict:
    docs = [doc for doc, _score in retrieved]
    scores = [score for _doc, score in retrieved]
    retrieved_issue_keys = {doc.metadata["issue_key"] for doc in docs}

    requested_issue_key_not_found = False
    if filters.get("issue_key"):
        requested_issue_key_not_found = filters["issue_key"] not in retrieved_issue_keys

    structured_keys = [k for k in ("assigned_team", "assignee", "priority", "status") if k in filters]
    requested_filters_not_reflected = False
    if structured_keys and docs:
        satisfied = any(all(doc.metadata.get(k) == filters[k] for k in structured_keys) for doc in docs)
        requested_filters_not_reflected = not satisfied

    return {
        "num_retrieved": len(docs),
        "zero_results": len(docs) == 0,
        "max_similarity": max(scores) if scores else 0.0,
        "requested_issue_key_not_found": requested_issue_key_not_found,
        "requested_filters_not_reflected": requested_filters_not_reflected,
        "ambiguous_assignee": bool(ambiguous_assignee_candidates),
        "llm_declined": llm_declined(answer),
        "unsupported_citations": unsupported_citations(answer, retrieved_issue_keys),
    }


def should_escalate(retrieved: list, filters: dict, ambiguous_assignee_candidates, answer: str) -> tuple[bool, list[str], dict]:
    evidence = evaluate_evidence(retrieved, filters, ambiguous_assignee_candidates, answer)
    reasons = []

    if evidence["zero_results"]:
        reasons.append("No relevant PowerOps records were retrieved for this question.")
    if evidence["requested_issue_key_not_found"]:
        reasons.append(f"Requested issue key '{filters.get('issue_key')}' was not found in the PowerOps knowledge base.")
    if evidence["requested_filters_not_reflected"]:
        reasons.append("No retrieved records match the requested team/assignee/priority/status combination.")
    if evidence["ambiguous_assignee"]:
        reasons.append(f"The assignee reference is ambiguous and matches multiple people: {ambiguous_assignee_candidates}.")
    if evidence["llm_declined"]:
        reasons.append("The language model indicated the retrieved context was insufficient to answer confidently.")
    if evidence["unsupported_citations"]:
        reasons.append(f"The generated answer cited Issue Key(s) not present in the retrieved evidence: {evidence['unsupported_citations']}.")

    return (len(reasons) > 0, reasons, evidence)


def load_escalations() -> list[dict]:
    if not ESCALATIONS_PATH.exists():
        return []
    with open(ESCALATIONS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_escalations(escalations: list[dict]) -> None:
    with open(ESCALATIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(escalations, f, indent=2)


def create_escalation(question: str, reasons: list[str], evidence: dict, retrieved_docs: list, user: str | None = None) -> dict:
    escalations = load_escalations()
    escalation_id = f"ESC-{len(escalations) + 1:04d}"
    retrieved_issue_ids = sorted({doc.metadata["issue_key"] for doc in retrieved_docs})

    record = {
        "escalation_id": escalation_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "user": user,
        "reason": "; ".join(reasons),
        "reasons": reasons,
        "retrieved_issue_ids": retrieved_issue_ids,
        "evidence": evidence,
        "status": "Pending Management Review",
    }
    escalations.append(record)
    save_escalations(escalations)
    return record


# ---------------------------------------------------------------------------
# ask_powerops — the full pipeline
# ---------------------------------------------------------------------------

def ask_powerops(question: str, vector_store, llm, vocabulary: dict, name_parts: dict, top_k: int = TOP_K) -> dict:
    retrieval = retrieve_powerops_documents(question, vector_store, vocabulary, name_parts, top_k=top_k)
    retrieved = retrieval["results"]
    docs = [doc for doc, _score in retrieved]
    filters = retrieval["filters"]
    ambiguous = retrieval["ambiguous_assignee_candidates"]
    sources = sorted({doc.metadata["issue_key"] for doc in docs})

    if not docs:
        answer = (
            "I could not find enough information in the PowerOps knowledge base "
            "to answer this question reliably. This request should be escalated "
            "to DevOps management."
        )
    else:
        context = build_context(docs)
        messages = [
            ("system", SYSTEM_PROMPT),
            ("human", f"PowerOps Context:\n\n{context}\n\nQuestion: {question}"),
        ]
        answer = llm.invoke(messages).content

    escalate, reasons, evidence = should_escalate(retrieved, filters, ambiguous, answer)

    escalation_record = None
    if escalate:
        escalation_record = create_escalation(question, reasons, evidence, retrieved_docs=docs)

    retrieved_details = [
        {
            "issue_key": doc.metadata["issue_key"],
            "assigned_team": doc.metadata["assigned_team"],
            "assignee": doc.metadata["assignee"],
            "priority": doc.metadata["priority"],
            "status": doc.metadata["status"],
            "score": round(score, 4),
        }
        for doc, score in retrieved
    ]

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "filters": filters,
        "semantic_query": retrieval["semantic_query"],
        "pinecone_filter": retrieval["pinecone_filter"],
        "ambiguous_assignee_candidates": ambiguous,
        "retrieved_details": retrieved_details,
        "retrieved_count": len(docs),
        "needs_escalation": escalate,
        "escalation": escalation_record,
    }


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="PowerOps", page_icon="🛠️", layout="centered")

st.title("PowerOps")
st.markdown("**AI-powered DevOps Management Assistant**")
st.caption(
    "Answers are grounded only in retrieved PowerOps records. When there isn't "
    "enough evidence to answer reliably, PowerOps escalates to management "
    "instead of guessing."
)

with st.sidebar:
    st.header("Example Questions")
    st.caption("Click one to fill the question box below.")
    for example in EXAMPLE_QUESTIONS:
        if st.button(example, key=f"example_{example}", use_container_width=True):
            st.session_state["question_input"] = example

try:
    vector_store, llm = get_clients()
    vocabulary = load_vocabulary()
    name_parts = {a: split_name(a) for a in vocabulary["assignees"]}
except Exception as exc:  # noqa: BLE001 - surface any startup failure clearly in the UI
    st.error(
        "PowerOps could not connect to its knowledge base. Check that `.env` "
        f"contains valid PINECONE_API_KEY / OPENAI_API_KEY.\n\nDetails: {exc}"
    )
    st.stop()

question = st.text_input(
    "Ask PowerOps a question about DevOps issues:",
    key="question_input",
    placeholder="e.g. Which high-priority problems belong to Nova Team?",
)

ask_clicked = st.button("Ask PowerOps", type="primary")

if ask_clicked and question.strip():
    with st.spinner("Retrieving PowerOps records and generating an answer..."):
        st.session_state["last_result"] = ask_powerops(question.strip(), vector_store, llm, vocabulary, name_parts)

result = st.session_state.get("last_result")

if result:
    st.subheader("Answer")
    st.markdown(result["answer"])

    if result["needs_escalation"]:
        esc = result["escalation"]
        st.error(
            "**Management Escalation Required**\n\n"
            "PowerOps could not find sufficient evidence to reliably answer this question.\n\n"
            f"**Escalation ID:** {esc['escalation_id']}\n\n"
            f"**Status:** {esc['status']}\n\n"
            f"**Reason:** {esc['reason']}"
        )
    else:
        st.success("Answered directly from the PowerOps knowledge base — no escalation needed.")

    with st.expander(f"Retrieved Sources ({result['retrieved_count']})"):
        if result["retrieved_details"]:
            st.dataframe(result["retrieved_details"], use_container_width=True, hide_index=True)
        else:
            st.write("No documents were retrieved for this question.")

    with st.expander("Applied Filters"):
        if result["filters"]:
            st.json(result["filters"])
        else:
            st.write("No structured filters were detected — this question was handled by semantic search alone.")
        if result["ambiguous_assignee_candidates"]:
            st.warning(f"Ambiguous assignee match: {result['ambiguous_assignee_candidates']}")

    with st.expander("Retrieval Details"):
        st.write("**Semantic query sent to Pinecone:**")
        st.code(result["semantic_query"])
        st.write("**Pinecone metadata filter applied:**")
        st.json(result["pinecone_filter"])
        st.write("**Source Issue Keys:**")
        st.write(", ".join(result["sources"]) if result["sources"] else "(none)")
elif ask_clicked:
    st.warning("Please enter a question.")
