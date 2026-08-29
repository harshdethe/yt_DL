"""
Next Word Predictor using LSTM (Keras/TensorFlow)
--------------------------------------------------
Same standard approach taught in CampusX "100 Days of Deep Learning"
Day 63 - LSTM Part 3: Next Word Predictor.

Steps:
1. Load a text corpus (replace `corpus` below with your own .txt file's content)
2. Tokenize the text and build a vocabulary
3. Create n-gram input sequences from every sentence
4. Pad sequences so they're all the same length
5. Split into X (all words except last) and y (the last word, one-hot encoded)
6. Build an Embedding -> LSTM -> Dense(softmax) model
7. Train it, then use it to predict the next word(s) for any input text

Install requirements first:
    pip install tensorflow numpy --break-system-packages
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.utils import to_categorical


# ---------------------------------------------------------------------------
# 1. CORPUS
# ---------------------------------------------------------------------------
# Replace this with your own text. For example, to load from a .txt file:
#   with open("faqs.txt", "r", encoding="utf-8") as f:
#       corpus = f.read()

corpus = """
The course follows a monthly subscription model where you have to pay
Rs 799 every month to continue with the course. But that also means
you don't have to pay Rs 799 for the next month if you don't want to.
The course covers Machine Learning, Deep Learning and MLOps. The teaching
mode is a mix of live sessions and recorded videos. Doubts are cleared
in the live sessions on weekends. There is no age limit or educational
background required for this course. Anyone who is interested in Data
Science can join this course.
"""


# ---------------------------------------------------------------------------
# 2. TOKENIZATION
# ---------------------------------------------------------------------------
tokenizer = Tokenizer()
tokenizer.fit_on_texts([corpus])

vocab_size = len(tokenizer.word_index) + 1  # +1 because indexing starts at 1
print("Vocabulary size:", vocab_size)


# ---------------------------------------------------------------------------
# 3. BUILD N-GRAM INPUT SEQUENCES
# ---------------------------------------------------------------------------
# For every sentence, generate progressively longer n-gram sequences:
#   "the course follows a" -> [the, course], [the, course, follows], [the, course, follows, a] ...
input_sequences = []
for sentence in corpus.split("\n"):
    if not sentence.strip():
        continue
    tokenized_sentence = tokenizer.texts_to_sequences([sentence])[0]
    for i in range(1, len(tokenized_sentence)):
        input_sequences.append(tokenized_sentence[: i + 1])

print("Number of training sequences:", len(input_sequences))


# ---------------------------------------------------------------------------
# 4. PAD SEQUENCES
# ---------------------------------------------------------------------------
max_len = max(len(seq) for seq in input_sequences)
padded_sequences = pad_sequences(input_sequences, maxlen=max_len, padding="pre")


# ---------------------------------------------------------------------------
# 5. SPLIT INTO X (input words) AND y (word to predict)
# ---------------------------------------------------------------------------
X = padded_sequences[:, :-1]
y = padded_sequences[:, -1]

# one-hot encode the target so the model can use categorical_crossentropy
y = to_categorical(y, num_classes=vocab_size)

print("X shape:", X.shape)
print("y shape:", y.shape)


# ---------------------------------------------------------------------------
# 6. BUILD THE MODEL: Embedding -> LSTM -> Dense(softmax)
# ---------------------------------------------------------------------------
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=100, input_length=max_len - 1),
    LSTM(150),
    Dense(vocab_size, activation="softmax"),
])

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()


# ---------------------------------------------------------------------------
# 7. TRAIN
# ---------------------------------------------------------------------------
model.fit(X, y, epochs=100, verbose=1)


# ---------------------------------------------------------------------------
# 8. PREDICT THE NEXT WORD(S)
# ---------------------------------------------------------------------------
def predict_next_words(seed_text, num_words=1):
    """Given a seed_text, predict and append `num_words` next words."""
    for _ in range(num_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_len - 1, padding="pre")

        predicted_probs = model.predict(token_list, verbose=0)
        predicted_index = np.argmax(predicted_probs, axis=-1)[0]

        # look up the word for the predicted index
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_index:
                output_word = word
                break

        seed_text += " " + output_word

    return seed_text


if __name__ == "__main__":
    print(predict_next_words("the course follows a", num_words=3))
    print(predict_next_words("data science can", num_words=5))
