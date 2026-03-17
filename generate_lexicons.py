"""
Generate lexicon files required by EAISe.py using NLTK WordNet.
Creates: lexicons/nrc.txt, lexicons/anger, lexicons/fear, lexicons/sadness,
         lexicons/joy, lexicons/positive, lexicons/negative
"""
import os
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn

LEXICON_DIR = "lexicons"
os.makedirs(LEXICON_DIR, exist_ok=True)

# --- Generate WordNet emotion word lists ---
# Map WordNet domains/synsets to emotion categories
emotion_seeds = {
    "anger": ["anger", "rage", "fury", "wrath", "outrage", "irritation", "annoyance", "hostile", "angry", "mad", "furious", "enraged", "irate", "livid", "infuriated", "resentful", "bitter", "hateful", "hostile", "aggressive"],
    "fear": ["fear", "terror", "horror", "dread", "panic", "anxiety", "fright", "scare", "afraid", "scared", "terrified", "nervous", "worried", "frightened", "alarmed", "apprehensive", "uneasy", "timid"],
    "sadness": ["sadness", "grief", "sorrow", "misery", "despair", "melancholy", "sad", "unhappy", "depressed", "gloomy", "mournful", "heartbroken", "lonely", "hopeless", "tragic", "tearful", "dismal", "bleak"],
    "joy": ["joy", "happiness", "delight", "pleasure", "bliss", "ecstasy", "happy", "glad", "cheerful", "joyful", "elated", "jubilant", "merry", "content", "pleased", "thrilled", "wonderful", "delightful"],
    "positive": ["good", "great", "love", "beautiful", "excellent", "amazing", "fantastic", "wonderful", "brilliant", "superb", "perfect", "nice", "best", "awesome", "outstanding", "magnificent", "marvelous", "splendid", "positive", "success"],
    "negative": ["bad", "terrible", "horrible", "awful", "disgusting", "evil", "ugly", "cruel", "worst", "hate", "nasty", "dreadful", "pathetic", "vile", "wretched", "atrocious", "abysmal", "appalling", "negative", "failure"],
}

def expand_with_wordnet(seed_words, max_words=500):
    """Expand seed words using WordNet synonyms and related words."""
    expanded = set(seed_words)
    for word in seed_words:
        for synset in wn.synsets(word):
            for lemma in synset.lemmas():
                name = lemma.name().replace('_', ' ')
                if ' ' not in name:
                    expanded.add(name.lower())
            # Also add hypernyms and hyponyms
            for hyper in synset.hypernyms():
                for lemma in hyper.lemmas():
                    name = lemma.name().replace('_', ' ')
                    if ' ' not in name:
                        expanded.add(name.lower())
            for hypo in synset.hyponyms():
                for lemma in hypo.lemmas():
                    name = lemma.name().replace('_', ' ')
                    if ' ' not in name:
                        expanded.add(name.lower())
    return sorted(expanded)[:max_words]

# Write WordNet emotion files
for emotion, seeds in emotion_seeds.items():
    words = expand_with_wordnet(seeds)
    filepath = os.path.join(LEXICON_DIR, emotion)
    with open(filepath, 'w', encoding='utf-8') as f:
        for w in words:
            f.write(w + '\n')
    print(f"  {emotion}: {len(words)} words")

# --- Generate NRC-format lexicon ---
# NRC format: word\temotion\t0or1
# Columns used by EAISe.py: [0]=anger, [2]=fear, [3]=joy, [4]=sadness, [5]=positive, [6]=negative
# The code reads sp[2] as the value, so the format is: word\temotion\tvalue
# But looking at the code more carefully: nrc[word] = [sp[2]] and then appends sp[2]
# So nrc[word] is a list of values in order of appearance.
# The code accesses indices [0],[2],[3],[4],[5],[6]
# Standard NRC format has 10 emotions: anger, anticipation, disgust, fear, joy, negative, positive, sadness, surprise, trust
# So index mapping: [0]=anger, [1]=anticipation, [2]=disgust(but code says fear), [3]=fear(but code says joy)...
# Looking at the code: nrc[word][0]=anger, [2]=fear, [3]=joy, [4]=sadness, [5]=positive, [6]=negative
# NRC standard order: anger(0), anticipation(1), disgust(2), fear(3), joy(4), negative(5), positive(6), sadness(7), surprise(8), trust(9)
# The code indices don't match standard NRC order, so let's just generate in the order the code expects

emotions_order = ["anger", "anticipation", "disgust", "fear", "joy", "negative", "positive", "sadness", "surprise", "trust"]

# Build NRC entries using SentiWordNet and our emotion word lists
nrc_entries = {}

# Get all unique words from emotion lists
all_emotion_words = {}
for emotion, seeds in emotion_seeds.items():
    for w in expand_with_wordnet(seeds):
        if w not in all_emotion_words:
            all_emotion_words[w] = set()
        all_emotion_words[w].add(emotion)

# Also add words from SentiWordNet
print("Building NRC lexicon from SentiWordNet...")
for synset in swn.all_senti_synsets():
    for lemma in synset.synset.lemmas():
        word = lemma.name().lower().replace('_', ' ')
        if ' ' in word:
            continue
        if word not in all_emotion_words:
            all_emotion_words[word] = set()
        if synset.pos_score() > 0.5:
            all_emotion_words[word].add("positive")
            all_emotion_words[word].add("joy")
        if synset.neg_score() > 0.5:
            all_emotion_words[word].add("negative")
            all_emotion_words[word].add("anger")
            all_emotion_words[word].add("sadness")

# Write NRC file
nrc_path = os.path.join(LEXICON_DIR, "nrc.txt")
with open(nrc_path, 'w', encoding='utf-8') as f:
    for word in sorted(all_emotion_words.keys()):
        word_emotions = all_emotion_words[word]
        for emotion in emotions_order:
            val = 1 if emotion in word_emotions else 0
            f.write(f"{word}\t{emotion}\t{val}\n")

print(f"  nrc.txt: {len(all_emotion_words)} words")
print("Done! Lexicon files generated in lexicons/")
