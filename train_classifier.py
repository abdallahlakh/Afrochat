import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Load data
data_dict = pickle.load(open('./data.pickle', 'rb'))
data = data_dict['data']
labels = np.asarray(data_dict['labels'])

# Determine the maximum sequence length
max_seq_length = max(len(seq) for seq in data)

# Pad sequences with zeros to have the same length
data_padded = np.array([np.pad(seq, (0, max_seq_length - len(seq)), 'constant') for seq in data])

# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(data_padded, labels, test_size=0.2, shuffle=True, stratify=labels)

# Train the model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Predict and calculate accuracy
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly !'.format(score * 100))

# Save the model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)