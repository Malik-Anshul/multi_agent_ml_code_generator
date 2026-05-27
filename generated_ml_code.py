# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import pandas as pd

# Load the iris dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Data preprocessing
def preprocess(df, target_column):
    # Drop any ID-like columns that have unique values for every row
    # These columns carry no predictive information
    for col in df.columns:
        if df[col].nunique() == len(df):
            df = df.drop(columns=[col])

    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Auto detect numerical and categorical columns
    numerical_cols = X.select_dtypes(include=["float64", "int64"]).columns
    categorical_cols = X.select_dtypes(include=["object"]).columns

    # Impute missing values
    # Numerical columns — fill with mean
    if len(numerical_cols) > 0:
        num_imputer = SimpleImputer(strategy="mean")
        X[numerical_cols] = num_imputer.fit_transform(X[numerical_cols])

    # Categorical columns — fill with most frequent value
    if len(categorical_cols) > 0:
        cat_imputer = SimpleImputer(strategy="most_frequent")
        X[categorical_cols] = cat_imputer.fit_transform(X[categorical_cols])

    # Encode target column
    # LabelEncoder for binary or ordinal target
    le = LabelEncoder()
    y = le.fit_transform(y)

    # Encode categorical features
    if len(categorical_cols) > 0:
        # Detect binary categorical columns — LabelEncoder
        binary_cols = [col for col in categorical_cols if X[col].nunique() == 2]
        multi_cols = [col for col in categorical_cols if X[col].nunique() > 2]

        # LabelEncoder for binary categorical columns
        for col in binary_cols:
            X[col] = le.fit_transform(X[col])

        # OneHotEncoder for multi category columns
        if len(multi_cols) > 0:
            ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
            encoded = ohe.fit_transform(X[multi_cols])
            encoded_df = pd.DataFrame(
                encoded,
                columns=ohe.get_feature_names_out(),
                index=X.index
            )
            # Drop original columns and concat encoded ones
            X = pd.concat([X.drop(columns=multi_cols), encoded_df], axis=1)

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale numerical features
    # fit_transform on train — only transform on test to prevent data leakage
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = preprocess(df, 'target')

# Reshape the data for CNN
X_train = X_train.reshape(-1, 4, 1)
X_test = X_test.reshape(-1, 4, 1)

# Build the CNN model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=2)
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(32 * 2, 64)
        self.fc2 = nn.Linear(64, 3)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.maxpool(x)
        x = x.view(-1, 32 * 2)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = Net()

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
for epoch in range(10):
    optimizer.zero_grad()
    outputs = model(torch.from_numpy(X_train).float())
    loss = criterion(outputs, torch.from_numpy(y_train).long())
    loss.backward()
    optimizer.step()
    print('Epoch {}: Loss = {:.4f}'.format(epoch+1, loss.item()))

# Evaluate the model
model.eval()
with torch.no_grad():
    outputs = model(torch.from_numpy(X_test).float())
    _, predicted = torch.max(outputs, 1)
    accuracy = (predicted == torch.from_numpy(y_test).long()).sum().item() / len(y_test)
    print('Test Accuracy: {:.2f}%'.format(100 * accuracy))

# Print the confusion matrix
print(confusion_matrix(y_test, predicted.numpy()))