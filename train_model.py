import pandas as pd
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

def kmer_featurization(sequences, k=6):
    kmers = []
    for seq in sequences:
        seq = str(seq).upper().replace("N", "")
        kmer_list = [seq[i:i+k] for i in range(len(seq) - k + 1)]
        kmers.append(' '.join(kmer_list))
    return kmers

def train_and_save_model():
    print("Loading data...")
    df1 = pd.read_csv('datafix.csv')
    
    # Check for columns with >50% missing values (as done in notebook)
    missing_ratio_df1 = df1.isnull().mean()
    drop_cols_df1 = missing_ratio_df1[missing_ratio_df1 > 0.5].index
    df1.drop(columns=drop_cols_df1, inplace=True)
    
    # Ensure df1 has at least 10,000 rows
    df1_subset = df1.head(10000).reset_index(drop=True)
    
    # Use df1_subset
    df = df1_subset.copy()
    
    print("Processing sequences...")
    # X = Epitope Sequence → k-mer
    X_seq = df['Epitope Sequence']
    X_kmers = kmer_featurization(X_seq, k=6)
    vectorizer = CountVectorizer()
    X_features = vectorizer.fit_transform(X_kmers)
    
    print("Preparing labels...")
    # y = Pivot Antibiotic + Resistant Phenotype
    # Convert "Resistant" to 1, "Susceptible" to 0
    df['Resistant Binary'] = df['Resistant Phenotype'].map({'Resistant': 1, 'Susceptible': 0})
    
    # Pivot: rows = sample, columns = Antibiotic, values = Resistant Binary
    y = df.pivot_table(index=df.index, columns='Antibiotic', values='Resistant Binary', fill_value=0)
    
    print(f"Shape of X: {X_features.shape}")
    print(f"Shape of y: {y.shape}")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.2, random_state=42)
    
    print("Training model...")
    # Model with class_weight
    base_model = RandomForestClassifier(n_estimators=100, class_weight='balanced')
    multi_model = MultiOutputClassifier(base_model)
    multi_model.fit(X_train, y_train)
    
    print("Saving model and components...")
    model_package = {
        'model': multi_model,
        'vectorizer': vectorizer,
        'antibiotic_columns': y.columns.tolist()
    }
    joblib.dump(model_package, 'model.pkl')
    
    print("Model training complete!")
    print("Model saved as: model.pkl")

if __name__ == "__main__":
    train_and_save_model()