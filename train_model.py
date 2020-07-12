# Necessary imports
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle
import json

# Load or JSON Configuration File
conf = json.load(open('conf.json'))

print("[EXEC] Loading face embeddings...")
data = pickle.loads(open(conf["embeddings"], "rb").read())

# Encode the labels
print("[EXEC] Encoding labels...")
le = LabelEncoder()
labels = le.fit_transform(data["names"])

print("[EXEC] Training model...")
recognizer = SVC(C=10.0, kernel="poly", degree=7, probability=True)
recognizer.fit(data["embeddings"], labels)

# Save the Face Recognizer Model
f = open(conf["recognizer"], "wb")
f.write(pickle.dumps(recognizer))
f.close()

# Save the Label encoder model
f = open(conf["le"], "wb")
f.write(pickle.dumps(le))
f.close()

#----------------- Run Command----------------
# python train_model.py