# Importing needed libraries 
import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

# Using a wrapper class for PCA, pixel difference to be passed along the SVM model
class PCASVMModel:
    # Initializing the model with pca 
    def __init__(self, pca, svm):
        self.pca = pca
        self.svm = svm

    def _extract_features(self, image_pairs):
        # Extracting the features by flattening the images and computing their pixel differences
        features = []
        for pair in image_pairs:
            img1, img2 = pair[0].flatten(), pair[1].flatten()
            diff = np.abs(img1 - img2)
            features.append(np.concatenate([img1, img2, diff]))
        return np.array(features)
    # Predict function 
    def predict(self, X):
        X = self._preprocess(X)
        X_pca = self.pca.transform(X)
        return self.svm.predict(X_pca)
    # Computing the score by first calling for the image pre-processing and predicting through the model 
    def score(self, X, y):
        X = self._preprocess(X)
        X_pca = self.pca.transform(X)
        y_pred = self.svm.predict(X_pca)
        return accuracy_score(y, y_pred)
    # Pre process function which reshapes data into pairs 
    def _preprocess(self, X_raw):
        X_pairs = X_raw.reshape(-1, 2, 62, 47) 
        return self._extract_features(X_pairs)

def extract_features(image_pairs):
    # Extracting the features by flattening the images and computing their pixel differences
    features = []
    for pair in image_pairs:
        img1, img2 = pair[0].flatten(), pair[1].flatten()
        diff = np.abs(img1 - img2)
        features.append(np.concatenate([img1, img2, diff]))
    return np.array(features)

def train_model(train_file, model_file):
    # Function to train the SVM model 
    # Load training data
    data = joblib.load(train_file)
    images = data['data'].reshape(-1, 2, 62, 47)  # Reshape into image pairs
    labels = data['target']

    # Extracting features
    X = extract_features(images)
    y = labels

    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=300)  # Retain 300 components
    X_pca = pca.fit_transform(X)

    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_pca, y, test_size=0.2, random_state=42)

    # Train SVM with Grid Search for optimal parameters
    svm = SVC(kernel='rbf')
    param_grid = {'C': [100], 'gamma': [0.1]}
    grid_search = GridSearchCV(svm, param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    # Evaluate on validation set
    best_svm = grid_search.best_estimator_
    y_pred = best_svm.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    print(f"Validation Accuracy: {acc:.4f}")

    # Save the wrapper model
    model = PCASVMModel(pca, best_svm)
    joblib.dump(model, model_file)
    print(f"Model saved in {model_file}")

if __name__ == "__main__":
    import sys
    train_file = sys.argv[1]
    model_file = sys.argv[2]
    train_model(train_file, model_file)