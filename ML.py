import pickle
import numpy

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Embedding, Bidirectional

sequence_length = 8000
embedding_dim = 300

model = Sequential()

model.add(Embedding(8000 + 1 , 32))
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(sequence_length, embedding_dim)))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))

model.add(LSTM(32, return_sequences=True))
model.add(LSTM(64))

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

nlp = spacy.load("en_core_web_md")

def analyze_review(text):
    text = nlp(text).vector
    text = np.reshape(text,(1,len(text)))
    result = model(text)
    
    
    return "Positive" if result>0.5 else "Negative"
