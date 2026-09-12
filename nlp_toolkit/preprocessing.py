from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer

def lowercase(text):
    return text.lower()

def remove_urls(text):
    clean_text = text
    in_urls = ['http', 'www', '.com', '.org', '.net']
    words = text.split()
    for word in words:
        for chunk in in_urls:
            if chunk in word:
                clean_text = clean_text.replace(word, '')
    return clean_text

def remove_mentions(text):
    clean_text = text.split()
    for i in range(len(clean_text)):
        if clean_text[i].startswith('@'):
            clean_text[i] = clean_text[i][1:]  # Remove the '@' symbol
    return ' '.join(clean_text)

#NOTE: this still breaks on instances of mixed case like "eBay" or "So ExCiTeD"
def handle_hashtags(text):
    clean_text = text.split()
    for i in range(len(clean_text)):
        if clean_text[i].startswith('#'):
            clean_text[i] = clean_text[i][1:]  # Remove the '#' symbol
            for j in range(len(clean_text[i])):
                if j > 0 and clean_text[i][j].isupper() and clean_text[i][j-1].islower(): 
                    clean_text[i] = clean_text[i][:j] + ' ' + clean_text[i][j:]  # Split camel case
    return ' '.join(clean_text)

#NOTE: run handle_hashtags() before this function to avoid removing the '#' symbol from hashtags
def remove_punctuation(text):
    punctuation = '''!()-[]{};:'"\\,<>./?#@$%^&*_~''' 
    clean_text = text.split()
    for i in range(len(clean_text)):
        for char in clean_text[i]:
            if char in punctuation:
                clean_text[i] = clean_text[i].replace(char, '')
    return ' '.join(clean_text)

def normalize_whitespace(text):
    return " ".join(text.split())

def tokenize(text):
    clean_text = text
    for i in range(len(text)):
        if i > 0 and clean_text[i] == '-' and clean_text[i-1].isalpha() and clean_text[i+1].isalpha():
            clean_text = clean_text[:i] + ' ' + clean_text[i+1:]  # Replace hyphen with space
    return clean_text.split()

def remove_stopwords(tokens, keep_negation=False):
    clean_tokens = tokens.copy()
    negation_words = {"no", "not", "nor", "never", "none", "nothing", "nowhere", "neither", "n't"}
    stop_words = stopwords.words('english')
    for word in negation_words:
        if word not in stop_words:
            stop_words.append(word)
    stop_words = set(stop_words)
    if keep_negation:
        stop_words = stop_words - negation_words
    return [t for t in clean_tokens if t.lower() not in stop_words]


def stem_tokens(tokens):
    stemmer = PorterStemmer()
    return [stemmer.stem(t) for t in tokens]

def get_wordnet_pos(tag):
    """Convert NLTK POS tag to WordNet POS tag."""
    if tag.startswith("J"):
        return wordnet.ADJ
    elif tag.startswith("V"):
        return wordnet.VERB
    elif tag.startswith("N"):
        return wordnet.NOUN
    elif tag.startswith("R"):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def lemmatize_tokens(tokens):
    lemmatizer = WordNetLemmatizer()
    pos_tags = pos_tag(tokens)
    converted_tags = [(word, get_wordnet_pos(tag)) for word, tag in pos_tags]
    return [lemmatizer.lemmatize(word, pos) for word, pos in converted_tags]