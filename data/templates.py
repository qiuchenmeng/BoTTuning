PROMPT_TEMPLATES = [

    # ============================================================
    # Group 1: Canonical / Baseline (标准参考模板)
    # ============================================================
    # 001
    """
        Imagine a bag containing small balls, each with a word written on it. {bag_description}
        You reach into the bag and randomly select one ball according to this exact probability distribution, just like drawing a physical ball from a real bag.
        Then predict which word is on the ball you selected.

        CRITICAL:
        - Select based SOLELY on the given frequency distribution.
        - Ignore all personal biases, preferences, word meanings, or prior knowledge about these words.
        - The selection must reflect ONLY the exact proportions provided.
        - If you output anything other than exactly one candidate word, the answer is invalid.

        MANDATORY OUTPUT FORMAT:
        - Output ONLY the single **English** word: {word_list}
        - No other text, explanations, punctuation, or characters.
        - Do not translate or modify the word.
        - The semantic meaning of the words in the bag is irrelevant.
    """,
    # 002
    """Imagine a bag containing small balls, each with a word written on it. {bag_description} Select a ball at random and predict its labeled word.

    CRITICAL:
    - Select based SOLELY on the given frequency distribution.
    - Ignore all personal biases, preferences, word meanings, or prior knowledge.
    - The selection must reflect ONLY the exact proportions provided.

    MANDATORY OUTPUT FORMAT:
    - Output ONLY the single English word: {word_list}
    - No other text, explanations, punctuation, or characters.

    """,

    # ============================================================
    # Group 2: Paraphrase / Rewriting (语义等价，语言重写)
    # ============================================================
    # 003
    """You have a bag of items, where each item is labeled with a word.
    Each word appears with the following probability distribution:
    {distribution}

    Your job is to randomly draw one item according to this distribution and then state which word it displays.

    Important:
    - The choice must strictly follow the probabilities.
    - Do not rely on word meanings or any personal judgment.
    - The meaning of the word implies nothing here.

    Answer format:
    - Return exactly one English word from: {word_list}
    - Do not add anything else.
    """,
    # 004
    """Suppose words are sampled from a discrete distribution given below:
    {distribution}

    One word is drawn at random following these exact probabilities.
    Please report the drawn word.

    Constraints:
    - Use only the numerical probabilities.
    - Ignore semantics and preferences.

    Output:
    - One word chosen from {word_list}
    """,

    # ============================================================
    # Group 3: Cut / Minimal / Compressed (缩写，信息最少)
    # ============================================================
    # 005
    """There is a bag containing small balls, each with a word written on it. {bag_description}

    Randomly draw a ball and predict the word written on it.

    Valid outputs is in {word_list}
    """,

    # 006
    """{distribution}

    Pick ONE word randomly from the bag.
    Output from {word_list}
    """,

    # ============================================================
    # Group 4: Add / Expanded / Redundant (扩写，加入冗余说明)
    # ============================================================
    # 007
    """You are a perfectly calibrated language model that produces tokens with probabilities that correspond to values in the context and prompt.
    I will give you a scenario.

    A set of words is placed into a container, where the likelihood of selecting
    each word is governed by the following probability distribution:
    {distribution}

    Although these words may carry meanings in natural language, you must
    completely ignore any semantic interpretation. The process is purely
    statistical and random.

    Your task is to simulate a single random draw from this distribution
    and report the selected word.

    Output rules:
    - Output exactly one word from: {word_list}
    - Do not include explanations, punctuation, or formatting.
    """,

    # 008
    """{bag_description}

    Imagine performing a physical random draw that perfectly respects this
    distribution and forecast the word written on it.

    Despite any intuition you may have, the outcome must depend ONLY on the numerical probabilities provided.

    Final answer requirements:
    - Return exactly one word from {word_list}
    - No additional text
    """,

    # ============================================================
    # Group 5: Mathematical / Formal (数学/统计表述)
    # ============================================================
    # 009
    """Let X be a categorical random variable with the following distribution:
    {distribution}

    Sample one realization of X and identify the word printed on it.

    Restrictions:
    - Sampling must follow the given probabilities exactly.
    - Semantic meaning is irrelevant.

    Return a randomly selected token from: {word_list}
    """,

    # 010
        """Given a discrete probability distribution over outcomes:
    {distribution}

    Draw one sample randomly from this distribution and predict which word is on the ball you sample.

    Output specification:
    - One English word from {word_list}
    - No other output
    """,

    # ============================================================
    # Group 6: Procedural / Programmatic (程序化 / 算法风格)
    # ============================================================
    # 011
    """Input:
    - Probability distribution (word -> probability):
    {distribution}

    Sample one word randomly and predict which word is on the ball you sample.

    Output one word from the following list:
    - {word_list}
    (single token)
    """,

    # 012
    """{bag_description}

    Task: stochastic sampling and return the word on the selected ball.

    Return exactly one of {word_list}
    """,

    # ============================================================
    # Group 7: Instruction-Weak / Prompt-Drop (刻意弱化指令)
    # ============================================================

    # 013
    """{bag_description}

    Predict that which word is randomly chosen.

    Answer:
    """,

    # 014
    """Words and probabilities:
    {distribution}

    Respond with one word that is randomly selected.
    """
    ]
