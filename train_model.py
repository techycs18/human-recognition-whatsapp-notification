# Necessary imports
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle
import json

print("[EXEC] Loading face embeddings...")
data = pickle.loads(open("output/embeddings.pickle", "rb").read())

# Encode the labels
print("[EXEC] Encoding labels...")
le = LabelEncoder()
labels = le.fit_transform(data["names"])

print("[EXEC] Training model...")
recognizer = SVC(C=10.0, kernel="poly", degree=7, probability=True)
recognizer.fit(data["embeddings"], labels)

# Save the Face Recognizer Model
f = open("output/recognizer.pickle", "wb")
f.write(pickle.dumps(recognizer))
f.close()

# Save the Label encoder model
f = open("output/le.pickle", "wb")
f.write(pickle.dumps(le))
f.close()

#----------------- Run Command----------------
# python train_model.py