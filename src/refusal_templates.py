"""
Shared refusal phrase pool for the Corrective-RAG training paradigm.

Used by:
  - build_sft_dataset.py  → picks a phrase (RNG-seeded) as the training target when
    the retriever did not surface the correct chunk (answer not in context).
  - score_corrective.py   → detects whether a generated answer is an abstention.
"""

REFUSAL_TEMPLATES = [
    "The provided context does not contain the answer to this question.",
    "I cannot answer this question based on the given information.",
    "The retrieved passages do not include the information needed to answer.",
    "Based on the context, the answer is not available.",
    "The information needed to answer is not present in the provided context.",
    "I could not find the answer in the given passages.",
    "The context does not provide enough information to answer this question.",
    "There is no answer to this question in the retrieved documents.",
    "The given information does not address this question.",
    "The answer cannot be determined from the provided context.",
    "None of the retrieved passages answer this question.",
    "The supplied context lacks the information required to answer.",
    "I am unable to answer this question from the given context.",
    "The retrieved information does not contain a relevant answer.",
    "This question cannot be answered using the provided passages.",
    "The context provided is not sufficient to answer the question.",
    "No relevant information was found in the given context.",
    "The answer is not contained in the retrieved chunks.",
    "I do not have enough information in the context to answer this.",
    "The passages provided do not mention the answer to this question.",
    "Unfortunately, the context does not hold the answer to this question.",
    "The information at hand does not allow me to answer the question.",
    "The provided documents do not contain what is needed to answer.",
    "I cannot find the requested information in the given context.",
    "The retrieved context is missing the answer to this question.",
    "There is insufficient information in the context to answer.",
    "The answer to this question is not present in the provided text.",
    "Based on the retrieved passages, I cannot provide an answer.",
    "The context does not include the details required to answer.",
    "I am not able to answer this from the information provided.",
    "The given passages are not relevant to answering this question.",
    "The answer is absent from the provided context.",
    "Nothing in the retrieved context answers this question.",
    "The supplied passages do not provide the answer.",
    "I cannot determine the answer from the available context.",
    "The provided information is insufficient to answer the question.",
    "The retrieved documents lack the answer to this question.",
    "This question is not addressed by the provided context.",
    "I could not locate the answer within the given passages.",
    "The context offered does not contain the relevant answer.",
    "No answer to this question appears in the provided context.",
    "The information provided does not cover this question.",
    "I have no basis in the given context to answer this question.",
    "The retrieved text does not supply the answer to this question.",
    "It is not possible to answer this question from the given context.",
    "The answer cannot be found in the supplied passages.",
    "The provided context fails to answer this question.",
    "I cannot respond to this question using the retrieved information.",
    "The needed information is missing from the provided passages.",
    "The context does not contain a valid answer to this question.",
]


REFUSAL_KEY_PHRASES = [
    "does not contain",
    "cannot answer",
    "can not answer",
    "could not find",
    "could not locate",
    "not contain the answer",
    "not available",
    "not present",
    "not contained",
    "cannot be determined",
    "cannot be found",
    "cannot find",
    "cannot provide an answer",
    "cannot determine",
    "do not contain",
    "does not include",
    "do not include",
    "does not provide",
    "do not provide",
    "does not mention",
    "do not mention",
    "not address",
    "does not address",
    "not enough information",
    "insufficient information",
    "not sufficient",
    "no relevant information",
    "no answer",
    "not present in the",
    "missing from",
    "unable to answer",
    "not able to answer",
    "is not present",
    "lacks the",
    "lack the",
    "not relevant",
    "no basis",
    "fails to answer",
    "not possible to answer",
]


def pick_refusal(rng):
    """Return one refusal phrase using the provided random. Random instance."""
    return rng.choice(REFUSAL_TEMPLATES)


def is_refusal_text(text: str) -> bool:
    """Heuristic: does a generated answer express abstention?

    True if it matches a template verbatim (normalized) or contains a key phrase.
    Deliberately lenient on the key-phrase side so paraphrased refusals still count.
    """
    if not text:
        return False
    low = text.strip().lower()
    # exact-ish match against the pool
    for t in REFUSAL_TEMPLATES:
        if low == t.lower() or low.rstrip(".") == t.lower().rstrip("."):
            return True
    # paraphrase-robust key-phrase match
    return any(kp in low for kp in REFUSAL_KEY_PHRASES)
