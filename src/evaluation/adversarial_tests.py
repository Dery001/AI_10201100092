# ============================================================
# adversarial_tests.py – Adversarial and edge-case test queries
# Student: [YOUR NAME] | Index: [YOUR INDEX NUMBER]
# ============================================================

ADVERSARIAL_QUERIES = [
    # Ambiguous
    {
        "id": "adv_01",
        "query": "Who won?",
        "type": "ambiguous",
        "expected_behaviour": "Should ask for clarification or return mixed results",
    },
    {
        "id": "adv_02",
        "query": "What is the result?",
        "type": "ambiguous",
        "expected_behaviour": "Retrieval may return irrelevant chunks due to vague wording",
    },
    # Misleading
    {
        "id": "adv_03",
        "query": "How much did Ghana spend on the 2024 election campaign?",
        "type": "misleading",
        "expected_behaviour": "Election data lacks campaign spending; should admit insufficient info",
    },
    {
        "id": "adv_04",
        "query": "What is Ghana's 2025 budget for military operations?",
        "type": "misleading",
        "expected_behaviour": "Budget document may not break down military; should say so",
    },
    # Incomplete
    {
        "id": "adv_05",
        "query": "votes",
        "type": "incomplete",
        "expected_behaviour": "Single keyword; retrieval uncertain; answer should be vague or insufficient",
    },
    {
        "id": "adv_06",
        "query": "allocation",
        "type": "incomplete",
        "expected_behaviour": "Too vague to return a precise answer",
    },
    # Out-of-scope
    {
        "id": "adv_07",
        "query": "What is the capital of France?",
        "type": "out_of_scope",
        "expected_behaviour": "Should return insufficient info message; no hallucination",
    },
    {
        "id": "adv_08",
        "query": "Explain quantum computing",
        "type": "out_of_scope",
        "expected_behaviour": "Clearly out of scope; no hallucination allowed",
    },
    # Factual (positive tests)
    {
        "id": "fact_01",
        "query": "What was the total voter turnout in the 2020 Ghana presidential election?",
        "type": "factual_election",
        "expected_behaviour": "Should retrieve from election CSV and cite specific numbers",
    },
    {
        "id": "fact_02",
        "query": "What is Ghana's total revenue for 2025 according to the budget?",
        "type": "factual_budget",
        "expected_behaviour": "Should retrieve budget chunk with GHc figures",
    },
    {
        "id": "fact_03",
        "query": "Which party won the most parliamentary seats in the 2020 election?",
        "type": "factual_election",
        "expected_behaviour": "Should name NPP or NDC with seat count",
    },
    {
        "id": "fact_04",
        "query": "What is Ghana's GDP growth target in the 2025 budget?",
        "type": "factual_budget",
        "expected_behaviour": "Should find GDP target percentage",
    },
]
