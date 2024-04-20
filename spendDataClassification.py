import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

dataAll = pd.read_csv('./dataIndexed.csv')
dataLabelled = pd.read_csv('./subsetDataLabelled.csv')
mapIDLabel = {}
for index, row in dataLabelled.iterrows():
    mapIDLabel[row['ID']] = row['Class']

for index,row in dataAll.iterrows():
	if row['ID'] in mapIDLabel:
		dataAll.at[index,'Class'] = mapIDLabel[row['ID']]

taxonomy = pd.read_csv('./taxonomy.csv')
classes = taxonomy[['Texonomy Level 1', 'Texonomy Level 2', 'Texonomy Level 3', 'Texonomy Level 4', 'Texonomy Level 5']].apply(lambda x: ' | '.join(x.dropna()), axis=1).str.lower()

labelEncoder = LabelEncoder()
labelEncoder.fit(classes)
# print(labelEncoder.classes_)

dataLabelled = dataAll.dropna(subset=['Class'])
dataLabelled['Class'] = dataLabelled['Class'].apply(lambda x: str(x).lower().lstrip(' \n.').rstrip(' \n.'))
dataUnlabelled = dataAll[dataAll['Class'].isna()]

X = []
y = labelEncoder.transform(dataLabelled['Class'])

# dataAll['encodedLabels'] = labelEncoder.transform(dataAll['Class'])
# dataAll['encodedLabels'].fillna(-1, inplace=True)

file = open("./openAIEmbeddingsNew.pkl",'rb')
embeddingMap = pickle.load(file)
file.close()


file = open("./openAIClassEmbeddingsNew.pkl",'rb')
classEmbeddingList = pickle.load(file)
file.close()

classEmbeddingMap = {}
for i,c in enumerate(classes):
	classEmbeddingMap[c] = i

for index,row in dataLabelled.iterrows():
	X.append(embeddingMap[row['ID']])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# print(labelEncoder.inverse_transform([3]))
text_embeddings_df = pd.DataFrame(X_test)
class_embeddings_df = pd.DataFrame( [ classEmbeddingList[i] for i,c in  enumerate(classes)])

print(text_embeddings_df.shape)
print(class_embeddings_df.shape)


similarity_matrix = cosine_similarity(text_embeddings_df, class_embeddings_df)

k = 3
top_k_classes_indices = np.argsort(similarity_matrix, axis=1)[:, -k:]
top_k_classes = [[classes[i] for i in indices[::-1]] for indices in top_k_classes_indices]

count = 0.0

for i in range(len(y_test.tolist())):
	if labelEncoder.inverse_transform([y_test[i]])[0] in top_k_classes[i]:
		count += 1
print(f"Recall@{k} VE : ", count/len(y_test))
print("Classification report VE : \n", classification_report(y_test, [labelEncoder.transform([e[0]])[0] for e in top_k_classes]))
print([labelEncoder.transform([e[0]])[0] for e in top_k_classes])

# smote = SMOTE(n_neighbors = 1)
# X_sm, y_sm = smote.fit_resample(X_train, y_train)


# rf_classifier = RandomForestClassifier(max_depth=1000, random_state=42, n_estimators=200)
rf_classifier = RandomForestClassifier(max_depth=400, random_state=42, n_estimators=50)
rf_classifier.fit(X_train, y_train)

# Predictions
predictionsRF = rf_classifier.predict(X_test)

# print(predictions[0])
# Evaluation

y_proba = rf_classifier.predict_proba(X_test)
top_k_classes_indices = np.argsort(y_proba, axis=1)[:, -k:]
top_k_classes = [rf_classifier.classes_[indices[::-1]] for indices in top_k_classes_indices]

count = 0.0
for i,c in enumerate(y_test):
	if c in top_k_classes[i]:
		count += 1
print(f"Recall@{k} RF : ",count/len(y_test))
print("Classification report RF : \n", classification_report(y_test, predictionsRF))

pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(X_test)
plt.scatter(reduced_embeddings[:,0], reduced_embeddings[:,1], c=y_test, cmap='rainbow')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('Training Data in Reduced Dimension')
plt.legend()
plt.show()

# print(X_sm, y_sm)
# # Train a semi-supervised classifier
# base_classifier = RandomForestClassifier()
# self_training_model = SelfTrainingClassifier(base_classifier, criterion='k_best', k_best=50)
# self_training_model.fit(X_sm, y_sm)
