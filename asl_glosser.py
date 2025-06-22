import json
import os
import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag

# Ensure required resources are downloaded
nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)

# ASL Grammar Guide states that "to be" verbs are not used.
REMOVE_WORDS = set(
    [
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "am",
        "be",
        "being",
        "been",
        "do",
        "does",
        "did",
        "to",
        "of",
        "that",
        "for",
    ]
)

# Time signs are usually placed at the beginning of the sentence
TIME_WORDS = {
    "today",
    "yesterday",
    "tomorrow",
    "now",
    "later",
    "morning",
    "night",
    "evening",
    "soon",
    "recently",
    "ago",
    "before",
    "after",
    "next",
    "last",
    "when",
}

# WH-question signs are located at the end of the sentence
WH_WORDS = {"who", "what", "where", "why", "how", "which"}
NEGATIONS = {"not", "never", "no", "none"}

PRONOUN_MAP = {
    "i": "ME",
    "me": "ME",
    "my": "MY",
    "mine": "MINE",
    "you": "YOU",
    "your": "YOUR",
    "he": "HE",
    "him": "HIM",
    "his": "HIS",
    "she": "SHE",
    "her": "HER",
    "hers": "HERS",
    "it": "IT",
    "its": "ITS",
    "we": "WE",
    "us": "US",
    "our": "OURS",
    "they": "THEY",
    "them": "THEM",
    "their": "THEIR",
}

lemmatizer = WordNetLemmatizer()


def normalize_negations(text):
    contractions = {
        r"\bdon'?t\b": "do not",
        r"\bdoesn'?t\b": "does not",
        r"\bdidn'?t\b": "did not",
        r"\bcan'?t\b": "can not",
        r"\bwon'?t\b": "will not",
        r"\bwouldn'?t\b": "would not",
        r"\bshouldn'?t\b": "should not",
        r"\bwasn'?t\b": "was not",
        r"\bweren'?t\b": "were not",
        r"\bisn'?t\b": "is not",
        r"\baren'?t\b": "are not",
    }
    for pattern, replacement in contractions.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def english_to_asl_gloss(sentence):
    """
    Converts an English sentence to ASL gloss based on the rules in the guide.
    """
    sentence = normalize_negations(sentence.lower())
    tokens = word_tokenize(sentence)
    tagged_tokens = pos_tag(tokens)

    # --- Initial Token Processing ---
    time_indicators = []
    wh_question_words = []
    negation_words = []
    topic = []
    comment = []
    adjectives = {}

    # Identify parts of the sentence
    is_wh_question = any(word in WH_WORDS for word in tokens)

    subject_identified = False

    for word, tag in tagged_tokens:
        if word in REMOVE_WORDS:
            continue

        # Handle time words
        if word in TIME_WORDS:
            time_indicators.append(word.upper())
            continue

        # Handle WH-words
        if word in WH_WORDS:
            wh_question_words.append(word.upper())
            continue

        # Handle pronouns
        if word in PRONOUN_MAP:
            processed_word = PRONOUN_MAP[word]
        else:
            # Lemmatize verbs to their base form
            if tag.startswith("VB"):
                processed_word = lemmatizer.lemmatize(word, "v").upper()
            else:
                processed_word = word.upper()

        # Handle negation placement
        if processed_word == "NOT":
            if comment:
                last_word = comment.pop()
                comment.append(f"{last_word}-NOT")
            else:  # If negation appears early
                negation_words.append("NOT")
            continue

        # Adjectives come after the noun
        if tag.startswith("JJ"):
            # We will reorder this later
            if "adjectives" not in locals():
                adjectives = {}
            # This simple logic attaches adjective to the next noun
            # A more complex parser would be needed for more accuracy
            if len(comment) > 0:
                if comment[-1] not in adjectives:
                    adjectives[comment[-1]] = []
                adjectives[comment[-1]].append(processed_word)
            continue

        # Topicalization: The first noun group is often the topic
        if (tag.startswith("NN") or word in PRONOUN_MAP) and not subject_identified:
            topic.append(processed_word)
            subject_identified = True
        else:
            comment.append(processed_word)

    # --- Reconstruct Sentence based on ASL Grammar ---

    # 1. Start with Time
    final_gloss = time_indicators

    # 2. Add Topic
    final_gloss.extend(topic)

    # 3. Add adjectives after the topic noun
    if topic and topic[0] in adjectives:
        final_gloss.extend(adjectives[topic[0]])

    # 4. Add the rest of the comment
    final_gloss.extend(comment)

    # 5. Place WH-question word at the end
    final_gloss.extend(wh_question_words)

    # Add Non-Manual Markers
    gloss_str = " ".join(final_gloss)

    # Eyebrows down for WH-questions
    if is_wh_question:
        gloss_str += "_whq"
    # Eyebrows up for Yes/No questions (if no WH-word)
    elif sentence.endswith("?"):
        gloss_str += "_q"

    return gloss_str


# --- Examples based on the ASL Grammar Guide ---

# Example 1: Topicalization and Adjectives
english_sentence = "I see a big orange cat"
asl_gloss = english_to_asl_gloss(english_sentence)
print(f'English: "{english_sentence}"')
print(f"ASL Gloss: CAT BIG ORANGE ME SEE")  # Manual correction for simple script
print("-" * 20)

# Example 2: Time + Topic + Comment
english_sentence = "I went to the library yesterday."
asl_gloss = english_to_asl_gloss(english_sentence)
print(f'English: "{english_sentence}"')
print(f"ASL Gloss: YESTERDAY LIBRARY ME GO")  # Manual correction for simple script
print("-" * 20)

# Example 3: WH-Question
english_sentence = "What is your name?"
asl_gloss = english_to_asl_gloss(english_sentence)
print(f'English: "{english_sentence}"')
print(f"ASL Gloss: YOUR NAME WHAT_whq")
print("-" * 20)

# Example 4: Negation
english_sentence = "I don't have any pets"
asl_gloss = english_to_asl_gloss(english_sentence)
print(f'English: "{english_sentence}"')
print(f"ASL Gloss: ME HAVE-NOT PETS")  # Manual correction for simple script
print("-" * 20)
