import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def load_and_process_data(file_path):
    
    df = pd.read_csv(file_path, sep=';', decimal=',', low_memory=False)

   
    df = df[df.columns[0]].str.split(',', expand=True)

    # Assign column names
    column_names = ['Date', 'Time', 'CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)',
                    'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)',
                    'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH']
    df.columns = column_names

    
    numeric_columns = df.columns[2:]  # All columns except Date and Time
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

    
    df = df.replace(-200, np.nan)

    print("Dataset shape:", df.shape)
    print("\ncolumn names:")
    print(df.columns)

    print("\nFirst few rows:")
    print(df.head())

    print("\ndata types:")
    print(df.dtypes)

    print("\nmissing values:")
    print(df.isnull().sum())

    # Visualize correlations
    plt.figure(figsize=(12, 10))
    sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=False, cmap='coolwarm')
    plt.title('Correlation Heatmap  Features')
    plt.show()

    return df

def preprocess_data(df):
    # Create target variable based on CO levels
    df['AirQuality_Class'] = pd.cut(df['CO(GT)'],
                            bins=[-np.inf, 0.5, 1.0, 1.5, 2.0, np.inf],
                            labels=[0, 1, 2, 3, 4])

    # Remove rows with NaN values in the target variable
    df = df.dropna(subset=['AirQuality_Class'])

    
    feature_columns = [col for col in df.columns if col not in ['Date', 'Time', 'AirQuality_Class', 'CO(GT)']]
    X = df[feature_columns]
    y = df['AirQuality_Class']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    
    print(f"\nTrain-Test Split Ratio:")
    print(f"Train data: {X_train.shape[0]} samples ({X_train.shape[0] / X.shape[0]:.2%})")
    print(f"Test data: {X_test.shape[0]} samples ({X_test.shape[0] / X.shape[0]:.2%})")

    
    class_names = {
        0: "Very Good (CO <= 0.5 ppm)",
        1: "Good (0.5 < CO <= 1.0 ppm)",
        2: "Moderate (1.0 < CO <= 1.5 ppm)",
        3: "Poor (1.5 < CO <= 2.0 ppm)",
        4: "Very Poor (CO > 2.0 ppm)"
    }
    print("\nClass Names and its descriptions:")
    for class_id, description in class_names.items():
        print(f"Class {class_id}: {description}")

   
    class_distribution = y.value_counts(normalize=True)
    print("\nClass Distributions:")
    for class_id, percentage in class_distribution.items():
        print(f"Class {class_id}: {percentage:.2%}")

    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
    X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

    # Scale the features
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    return X_train, X_test, y_train, y_test

def train_and_evaluate_model(X_train, X_test, y_train, y_test):
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
        'SVM': SVC(kernel='rbf', random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Naive Bayes': GaussianNB()
    }

    results = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy

        print(f"{name} Classifier Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        # Plot confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {name} Classifier')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()

        # Feature importance (for models that support it)
        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.DataFrame({'feature': X_train.columns, 'importance': model.feature_importances_})
            feature_importance = feature_importance.sort_values('importance', ascending=False)

            plt.figure(figsize=(12, 8))
            sns.barplot(x='importance', y='feature', data=feature_importance)
            plt.title(f'Feature Importance - {name} Classifier')
            plt.show()

    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(results.keys()), y=list(results.values()))
    plt.title('Model Comparisons')
    plt.xlabel('Models')
    plt.ylabel('Accuracy of the model')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def main():
    # Load and process data
    df = load_and_process_data('AirQualityUCI.csv')

    # Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # Train and evaluate models
    train_and_evaluate_model(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()
