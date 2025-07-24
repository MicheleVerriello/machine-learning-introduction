from autoimmune_diseases_dataset import AutoimmuneDiseasesDataset
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

# 1. Load the dataset
dataset = AutoimmuneDiseasesDataset('../data/Final_Balanced_Autoimmune_Disorder_Dataset.csv')
X = dataset.X
y = dataset.y

# 2. Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Create a pipeline: scaling + neural network
model = make_pipeline(
    StandardScaler(),
    MLPClassifier(hidden_layer_sizes=(200, 100, 50, 25), max_iter=500, random_state=42, alpha=0.01)
)
model.fit(X_train, y_train)

# 4. Evaluate the model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# 5. Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')

print("Cross-validated scores:", scores)
print("Mean accuracy:", scores.mean())