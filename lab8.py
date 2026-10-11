import pandas as pd
import glob
import os
import re
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import LatentDirichletAllocation
from nltk.stem import PorterStemmer

# Load news articles
folder = "news-articles"
files = glob.glob(os.path.join(folder, "*.txt"))

texts = []

for file in files:
    with open(file, "r", encoding="utf-8") as f:
        texts.append(f.read())

df = pd.DataFrame({
    "Filename": [os.path.basename(f) for f in files],
    "Text": texts
})

# Text preprocessing
stemmer = PorterStemmer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    words = text.split()
    words = [
        stemmer.stem(w)
        for w in words
        if w not in ENGLISH_STOP_WORDS
    ]
    return " ".join(words)

df["Cleaned"] = df["Text"].apply(preprocess)

# Create Document-Term Matrix
vectorizer = CountVectorizer(max_df=0.95, min_df=1)
X = vectorizer.fit_transform(df["Cleaned"])
terms = vectorizer.get_feature_names_out()

# Frequent terms
freq = X.sum(axis=0).A1

freq_df = pd.DataFrame({
    "Word": terms,
    "Frequency": freq
}).sort_values(by="Frequency", ascending=False)

print("Top Frequent Terms")
print(freq_df.head(10))

# Plot frequent terms
freq_df.head(10).plot.bar(
    x="Word",
    y="Frequency",
    legend=False,
    figsize=(8, 4)
)

plt.ylabel("Frequency")
plt.show()

# LDA Topic Modelling
lda = LatentDirichletAllocation(
    n_components=5,
    random_state=42
)

lda.fit(X)

print("\nHidden Topics")

for i, topic in enumerate(lda.components_):
    print(f"\nTopic {i + 1}")
    print([terms[j] for j in topic.argsort()[-10:][::-1]])

# Document Topic Distribution
doc_topics = lda.transform(X)

result = pd.DataFrame(
    doc_topics,
    columns=[f"Topic {i + 1}" for i in range(5)]
)

result["Dominant Topic"] = doc_topics.argmax(axis=1) + 1
result.insert(0, "Filename", df["Filename"])

