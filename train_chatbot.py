import json, random, nltk, numpy as np
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD

nltk.download('punkt')    # tonkenizer
nltk.download('punkt_tab')
nltk.download('wordnet')  # Lemmatization

lemmatizer = WordNetLemmatizer()
intents = json.load(open('intents.json'))


words = []
classes = []
documents = []

for intent in intents['intents']:
    for pattern in intent['patterns']:
        # tokenize each word in the sentence
        tokens = nltk.word_tokenize(pattern)
        words.extend(tokens)
        
        # add to documents in the corpus
        documents.append((tokens, intent['tag']))
        
        # add to classes if not already there
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

    
words = [lemmatizer.lemmatize(w.lower()) for w in words if w.isalpha()]
words = sorted(set(words))
classes = sorted(set(classes))

training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    token_words = [lemmatizer.lemmatize(w.lower()) for w in doc[0]]
    for w in words:
        bag.append(1 if w in token_words else 0)

    output_row = output_empty[:]
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])


random.shuffle(training)
train_x = np.array([x[0] for x in training])
train_y = np.array([x[1] for x in training])

model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5')

# Save words and classes
import pickle
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))