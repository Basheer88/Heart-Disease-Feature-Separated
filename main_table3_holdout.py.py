import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
import shap
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

RUN_CV = True          # True: 10-fold CV, False: single split
N_SPLITS = 10          # set to 10 (or 5 if you want)

class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.best_score = None
        self.epochs_no_improve = 0
        self.early_stop = False

    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.epochs_no_improve += 1
            if self.epochs_no_improve >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.epochs_no_improve = 0
        if self.early_stop and self.verbose:
            print("Early stopping triggered")

# Load dataset
df = pd.read_csv('data/HeartCT.csv')

# Define features for both channels
features_channel_1 = ['age', 'sex', 'chol', 'trestbps', 'fbs']
features_channel_2 = ['cp', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
target = 'target'

# Split dataset
X1 = df[features_channel_1].values
X2 = df[features_channel_2].values
y = df[target].values

# Train-test split for both channels
X_train_1, X_test_1, y_train, y_test = train_test_split(X1, y, test_size=0.2, random_state=42)
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X2, y, test_size=0.2, random_state=42)

# Adjusting the training set for Channel 1 to match the size used for Channel 2's DNN training
X_train_1_aligned, X_temp_1, y_train_aligned, y_temp = train_test_split(X_train_1, y_train, test_size=0.2, random_state=42)

# Further split training data of Channel 2 into training and validation sets
X_train_2, X_val_2, y_train_2, y_val_2 = train_test_split(X_train_2, y_train_2, test_size=0.2, random_state=42)

# Standardize the aligned training set for Channel 1
scaler_1_aligned = StandardScaler().fit(X_train_1_aligned)
X_train_1_aligned = scaler_1_aligned.transform(X_train_1_aligned)
X_test_1 = scaler_1_aligned.transform(X_test_1) 

# Standardize features
scaler_1 = StandardScaler().fit(X_train_1)
X_train_1 = scaler_1.transform(X_train_1)
#X_test_1 = scaler_1.transform(X_test_1)

scaler_2 = StandardScaler().fit(X_train_2)
X_train_2 = scaler_2.transform(X_train_2)
X_val_2 = scaler_2.transform(X_val_2)
X_test_2 = scaler_2.transform(X_test_2)

# Convert to PyTorch tensors
X_train_2_tensor = torch.tensor(X_train_2, dtype=torch.float32)
X_val_2_tensor = torch.tensor(X_val_2, dtype=torch.float32)
X_test_2_tensor = torch.tensor(X_test_2, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_2, dtype=torch.float32).view(-1, 1)
y_val_tensor = torch.tensor(y_val_2, dtype=torch.float32).view(-1, 1)




# Channel 1: Ensemble XGBoost with SVM
model_xgb = xgb.XGBClassifier(learning_rate= 0.25, max_depth= 3, n_estimators= 500, subsample= 0.8, eval_metric='logloss')
model_svm = SVC(C=100, gamma=100, kernel='rbf', probability=True)  # Ensure probability=True for soft voting

# Combine into a VotingClassifier for the ensemble
ensemble_model = VotingClassifier(estimators=[('xgb', model_xgb), ('svm', model_svm)], voting='soft')

# Train the ensemble model on the aligned training set
ensemble_model.fit(X_train_1_aligned, y_train_aligned)

# Evaluate ensemble
ensemble_predictions = ensemble_model.predict(X_test_1)
ensemble_accuracy = accuracy_score(y_test, ensemble_predictions)
print(f"Accuracy of Ensemble model (Channel 1): {ensemble_accuracy * 100:.2f}%")


# Generating predictions for the train set from the ensemble model on the aligned training set
ensemble_train_pred = ensemble_model.predict_proba(X_train_1_aligned)[:, 1]

# Generating predictions for the test set from the ensemble model
ensemble_test_pred = ensemble_model.predict_proba(X_test_1)[:, 1]




# Fit the explainer
explainer_xgb = shap.explainers.Tree(ensemble_model.named_estimators_['xgb'], X_train_1_aligned)
shap_values_xgb = explainer_xgb.shap_values(X_test_1)

# Summary plot for Channel 1
shap.summary_plot(shap_values_xgb, X_test_1, feature_names=features_channel_1)

# ROC AUC
# Calculate predicted probabilities for the positive class for each model
xgb_probs = ensemble_model.named_estimators_['xgb'].predict_proba(X_test_1)[:, 1]
svm_probs = ensemble_model.named_estimators_['svm'].predict_proba(X_test_1)[:, 1]
ensemble_probs = ensemble_model.predict_proba(X_test_1)[:, 1]

# Calculate ROC curve and ROC area for each model
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_probs)
roc_auc_xgb = roc_auc_score(y_test, xgb_probs)

fpr_svm, tpr_svm, _ = roc_curve(y_test, svm_probs)
roc_auc_svm = roc_auc_score(y_test, svm_probs)

fpr_ensemble, tpr_ensemble, _ = roc_curve(y_test, ensemble_probs)
roc_auc_ensemble = roc_auc_score(y_test, ensemble_probs)

# Plotting all ROC curves
plt.figure(figsize=(10, 8))
plt.plot(fpr_xgb, tpr_xgb, color='blue', lw=2, label='XGBoost (area = %0.2f)' % roc_auc_xgb)
plt.plot(fpr_svm, tpr_svm, color='green', lw=2, label='SVM (area = %0.2f)' % roc_auc_svm)
plt.plot(fpr_ensemble, tpr_ensemble, color='darkorange', lw=2, label='Ensemble (area = %0.2f)' % roc_auc_ensemble)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) - Ensemble and Individual Models')
plt.legend(loc="lower right")
plt.show()


# Channel 2: DNN with PyTorch (with Attention Mechanism and L2 Regularization)
class Attention(nn.Module):
    def __init__(self, feature_dim):
        super(Attention, self).__init__()
        self.attention_weights = nn.Parameter(torch.randn(feature_dim))

    def forward(self, x):
        # Assuming x is [batch_size, feature_dim]
        attention_scores = x @ self.attention_weights  # Simplified dot product
        # Reshape attention scores to [batch_size, 1] for softmax
        attention_scores = attention_scores.unsqueeze(-1)
        # Now applying softmax across each feature (now at dim=1 after unsqueeze)
        attention_scores = torch.softmax(attention_scores, dim=1)
        # Squeeze back to [batch_size] to weight the features
        attention_scores = attention_scores.squeeze(-1)
        # Weighting features by attention scores (broadcasting)
        attended_features = x * attention_scores.unsqueeze(-1)
        return attended_features
    
class DNNWithAttention(nn.Module):
    def __init__(self, input_dim, n_units):
        super(DNNWithAttention, self).__init__()
        self.attention = Attention(input_dim)
        self.fc1 = nn.Linear(input_dim, n_units)
        self.fc2 = nn.Linear(n_units, 1)

    def forward(self, x):
        x = self.attention(x)
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x


model_dnn = DNNWithAttention(X_train_2.shape[1], 256)
criterion = nn.BCELoss()
optimizer = optim.Adam(model_dnn.parameters(), lr=0.01, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5)
early_stopping = EarlyStopping(patience=20)

# Creating TensorDatasets for Channel 2
train_dataset_2 = TensorDataset(X_train_2_tensor, y_train_tensor)
val_dataset_2 = TensorDataset(X_val_2_tensor, y_val_tensor)

# batch size
batch_size = 32 

# Creating DataLoaders
train_loader_2 = DataLoader(train_dataset_2, batch_size=batch_size, shuffle=True)
val_loader_2 = DataLoader(val_dataset_2, batch_size=batch_size, shuffle=False)



training_losses = []
validation_losses = []
training_accuracies = []
validation_accuracies = []
# Training loop with early stopping for the DNN model
for epoch in range(500):
    # Training phase
    model_dnn.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    for inputs, labels in train_loader_2:
        optimizer.zero_grad()
        outputs = model_dnn(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        predicted = (outputs> 0.5).float()
        train_correct += (predicted == labels).sum().item()
        train_total += labels.size(0)
        

    train_accuracy = (train_correct / train_total) * 100
    training_losses.append(train_loss / len(train_loader_2))
    training_accuracies.append(train_accuracy)

    # Validation phase
    model_dnn.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for inputs, labels in val_loader_2:
            outputs = model_dnn(inputs)
            val_loss += criterion(outputs, labels).item()

            predicted = (outputs > 0.5).float()
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)

    val_accuracy = (val_correct / val_total) * 100
    validation_losses.append(val_loss / len(val_loader_2))
    validation_accuracies.append(val_accuracy)

    print(f'Epoch {epoch+1}, Train Loss: {train_loss / len(train_loader_2):.4f}, Val Loss: {val_loss / len(val_loader_2):.4f}, Train Acc: {train_accuracy:.2f}%, Val Acc: {val_accuracy:.2f}%')

    #print(f'Epoch {epoch}, Loss: {loss.item()}, Val Loss: {val_loss.item()}')
    scheduler.step(val_loss)
    early_stopping(val_loss)
    

    if early_stopping.early_stop:
        print("Early stopping")
        break


# plotting
epochs = range(1, len(training_losses) + 1)  # training_losses stores loss per epoch

# Plotting training and validation loss
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, training_losses, 'b-', label='Training Loss')
plt.plot(epochs, validation_losses, 'r-', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plotting training and validation accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, training_accuracies, 'b-', label='Training Accuracy')
plt.plot(epochs, validation_accuracies, 'r-', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()


# tensors for both training and testing data for Channel 2
X_train_2_tensor = torch.tensor(X_train_2, dtype=torch.float32)
X_test_2_tensor = torch.tensor(X_test_2, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)


# Evaluation for Channel 2
with torch.no_grad():
    model_dnn.eval()  # Set the model to evaluation mode
    dnn_outputs = model_dnn(X_test_2_tensor).squeeze()
    dnn_predictions = torch.round(dnn_outputs)
    dnn_accuracy = (dnn_predictions == y_test_tensor).float().mean()
    print(f"Accuracy of DNN model (Channel 2) with Attention Mechanism: {dnn_accuracy.item() * 100:.2f}%")


# Generating DNN predictions for the aligned training subset
with torch.no_grad():
    model_dnn.eval()
    dnn_train_pred = model_dnn(torch.tensor(X_train_2, dtype=torch.float32)).squeeze().numpy()
    # Test predictions remain the same
    dnn_test_pred = model_dnn(X_test_2_tensor).squeeze().numpy()


background_data = X_train_2_tensor[:100]  # Use a subset of training data for efficiency

# Instantiate the explainer with the background dataset
explainer_dnn = shap.GradientExplainer(model_dnn, background_data)

# Generate SHAP values for a subset of test examples
shap_values_dnn, expected_value_dnn = explainer_dnn.shap_values(X_test_2_tensor[:10], ranked_outputs=1)


# For a simple summary plot based on generated SHAP values
correct_shape_shap_values = shap_values_dnn.reshape(-1, X_test_2_tensor.shape[1])  # Example reshaping
print(correct_shape_shap_values.shape)
print(X_test_2_tensor[:8].numpy().shape)
# Verify the length of the features list matches the number of features in the SHAP values and data
print(len(features_channel_2))


# Make sure the SHAP values correspond to the instances being plotted
print("Shapes before plot:", correct_shape_shap_values.shape, X_test_2_tensor[:8].numpy().shape)
shap.summary_plot(correct_shape_shap_values[:8], X_test_2_tensor[:8].numpy(), feature_names=features_channel_2)




# Fuse the training predictions
X_train_fused = np.column_stack((ensemble_train_pred, dnn_train_pred))
X_train_fused_tensor = torch.tensor(X_train_fused, dtype=torch.float32)

# Fuse the testing predictions
X_test_fused = np.column_stack((ensemble_test_pred, dnn_test_pred))
X_test_fused_tensor = torch.tensor(X_test_fused, dtype=torch.float32)

# Ensure that both predictions are from the test sets
assert ensemble_test_pred.shape[0] == dnn_test_pred.shape[0], "Mismatch in prediction sizes for fusion."


y_train_tensor = torch.tensor(y_train_aligned, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Fusion and Prediction Layer
class FusionNN(nn.Module):
    def __init__(self):
        super(FusionNN, self).__init__()
        self.fc1 = nn.Linear(2, 64)  # 2 inputs: one from each channel
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc2(x))
        return x



# Prepare the fused data
# Preparing fused data for the test set
X_test_fused = np.column_stack((ensemble_test_pred, dnn_test_pred))
X_test_fused_tensor = torch.tensor(X_test_fused, dtype=torch.float32)


X_train_fused = np.column_stack((ensemble_train_pred, dnn_train_pred))
X_train_fused_tensor = torch.tensor(X_train_fused, dtype=torch.float32)


y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Initialize the Fusion Model
fusion_model = FusionNN()
optimizer = optim.Adam(fusion_model.parameters(), lr=0.001)

y_train_tensor = torch.tensor(y_train_aligned, dtype=torch.float32).view(-1, 1)





# Training loop for the Fusion Model
for epoch in range(300):
    optimizer.zero_grad()
    outputs = fusion_model(X_train_fused_tensor).squeeze()
    loss = criterion(outputs, y_train_tensor.squeeze()) 
    loss.backward()
    optimizer.step()


# Evaluation on the test set
with torch.no_grad():
    fusion_model.eval()  # set the model in evaluation mode
    predictions = fusion_model(X_test_fused_tensor).squeeze().round()
    accuracy = (predictions == y_test_tensor).float().mean()
    print(f'Accuracy on the test set: {accuracy.item() * 100:.2f}%')

# Convert predictions and true labels to NumPy arrays for metric calculation if they are in tensors
predictions_np = predictions.numpy()
y_test_np = y_test_tensor.numpy()



# Calculate precision, recall, and F1-score
precision = precision_score(y_test_np, predictions_np)
recall = recall_score(y_test_np, predictions_np)
f1 = f1_score(y_test_np, predictions_np)

# Print the metrics
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')





# Ensure X_test_fused is a numpy array for LIME
if not isinstance(X_test_fused, np.ndarray):
    X_test_fused = X_test_fused.numpy()  # Convert to numpy array if it's a tensor

def fusion_model_predict_proba(X):
    # Convert X to a tensor if it's not already
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X, dtype=torch.float32)
    
    # Ensure the model is in evaluation mode
    fusion_model.eval()
    
    with torch.no_grad():
        # Forward pass
        predictions = fusion_model(X)
        
        # Convert predictions to probabilities (binary classification with sigmoid output)
        probabilities = torch.sigmoid(predictions).cpu().numpy()
        
        # LIME expects probabilities for each class (binary: [class 0, class 1])
        probabilities = np.hstack([1 - probabilities, probabilities])
        return probabilities



# Feature names for LIME (representing predictions from Channel 1 and Channel 2)
features_fusion = ['Prediction from Channel 1', 'Prediction from Channel 2']

# Initialize LIME explainer
explainer_fusion = LimeTabularExplainer(X_train_fused, 
                                        feature_names=features_fusion, 
                                        class_names=['No Heart Disease', 'Heart Disease'], 
                                        discretize_continuous=True)

# Select a random instance to explain
i = np.random.randint(0, X_test_fused.shape[0])

# Explain the instance
exp = explainer_fusion.explain_instance(X_test_fused[i], fusion_model_predict_proba, num_features=2)

# Display the explanation
#exp.show_in_notebook(show_table=True, show_all=False)
exp.save_to_file('lime_explanation.html')

# Get the textual explanation for an instance
explanation_text = exp.as_list()

# Print or save the explanation text to a file
print(explanation_text)
with open('lime_explanation.txt', 'w') as file:
    for feature, weight in explanation_text:
        file.write(f'{feature}: {weight}\n')

import matplotlib.pyplot as plt

# 'exp' is our LIME explanation and extracted 'explanation_text'
features, weights = zip(*exp.as_list())
plt.barh(features, weights)
plt.xlabel('Feature Contribution')
plt.title('LIME Explanation')
plt.tight_layout()
plt.savefig('lime_explanation.png')
plt.show()


