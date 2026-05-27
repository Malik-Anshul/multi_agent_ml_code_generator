class PreprocessingTemplates:

    # ===========================================================
    # REGISTRY — maps category to template name
    # ===========================================================
    registry = {
        "supervised": "supervised_preprocessing",
        "nlp": "nlp_preprocessing",
        "cnn": "cnn_preprocessing"
    }

    # ===========================================================
    # SUPERVISED PREPROCESSING TEMPLATE
    # ===========================================================
    supervised_preprocessing = """
        import pandas as pd
        import numpy as np
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
        from sklearn.impute import SimpleImputer


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
        """

    # ===========================================================
    # NLP PREPROCESSING TEMPLATE
    # ===========================================================
    nlp_preprocessing = """
        import pandas as pd
        import numpy as np
        import re
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder
        import nltk
        from nltk.tokenize import word_tokenize
        from nltk.corpus import stopwords
        from nltk.stem import PorterStemmer

        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
        nltk.download("stopwords", quiet=True)


        def _clean_text(text):
            # Convert to lowercase — normalize all text
            text = text.lower()

            # Remove URLs — not useful for language understanding
            text = re.sub(r"http\S+", "", text)

            # Remove HTML tags — clean web scraped data
            text = re.sub(r"<.*?>", "", text)

            # Remove punctuation and special characters
            text = re.sub(r"[^A-Za-z0-9\s]", "", text)

            return text


        def _remove_stopwords(text):
            # Tokenize and remove common words that carry no meaning
            tokens = word_tokenize(text)
            stop_words = stopwords.words("english")
            filtered = [word for word in tokens if word not in stop_words]
            return " ".join(filtered)


        def _stemming(text):
            # Reduce words to root form — running, runs, ran -> run
            ps = PorterStemmer()
            tokens = word_tokenize(text)
            stemmed = [ps.stem(token) for token in tokens]
            return " ".join(stemmed)


        def _build_vocab(texts):
            # Build vocabulary from training texts
            # Index 0 reserved for padding
            vocab = {}
            idx = 1
            for sentence in texts:
                for word in sentence.split():
                    if word not in vocab:
                        vocab[word] = idx
                        idx += 1
            return vocab


        def _encode_sentence(sentence, vocab, max_len=200):
            # Convert sentence to sequence of indices
            tokens = sentence.split()
            encoded = [vocab.get(word, 0) for word in tokens]

            # Pad or truncate to fixed length
            if len(encoded) < max_len:
                encoded += [0] * (max_len - len(encoded))
            else:
                encoded = encoded[:max_len]
            return encoded


        def preprocess(df, text_column, target_column, batch_size=64, max_len=200):
            # Remove duplicates
            df = df.drop_duplicates()

            # Drop rows with null values in text or target
            df = df.dropna(subset=[text_column, target_column])

            # Apply full text cleaning pipeline
            df[text_column] = df[text_column].apply(_clean_text)
            df[text_column] = df[text_column].apply(_remove_stopwords)
            df[text_column] = df[text_column].apply(_stemming)

            # Encode target labels to numerical values
            le = LabelEncoder()
            df[target_column] = le.fit_transform(df[target_column])

            X = df[text_column]
            y = df[target_column]

            # Train test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Build vocabulary from training data only
            # Never build vocab from test data — prevents data leakage
            vocab = _build_vocab(X_train)

            # Encode sentences to index sequences
            X_train_enc = np.array([_encode_sentence(s, vocab, max_len) for s in X_train])
            X_test_enc = np.array([_encode_sentence(s, vocab, max_len) for s in X_test])

            # Convert to PyTorch tensors
            train_dataset = TensorDataset(
                torch.tensor(X_train_enc, dtype=torch.long),
                torch.tensor(y_train.values, dtype=torch.float)
            )
            test_dataset = TensorDataset(
                torch.tensor(X_test_enc, dtype=torch.long),
                torch.tensor(y_test.values, dtype=torch.float)
            )

            # Create DataLoaders for batch training
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            return train_loader, test_loader, vocab
        """

    
    cnn_preprocessing = """
        import torch
        import torchvision
        import torchvision.transforms as transforms
        from torch.utils.data import DataLoader


        def preprocess(dataset_name="mnist", batch_size=64):
            # Define transforms — normalize pixel values to [-1, 1]
            # Helps model converge faster during training
            if dataset_name == "mnist":
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                ])

                # Load MNIST dataset — handwritten digit classification
                train_dataset = torchvision.datasets.MNIST(
                    root="./data", train=True, download=True, transform=transform
                )
                test_dataset = torchvision.datasets.MNIST(
                    root="./data", train=False, download=True, transform=transform
                )

            elif dataset_name == "cifar10":
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])

                # Load CIFAR10 dataset — 10 class image classification
                train_dataset = torchvision.datasets.CIFAR10(
                    root="./data", train=True, download=True, transform=transform
                )
                test_dataset = torchvision.datasets.CIFAR10(
                    root="./data", train=False, download=True, transform=transform
                )

            # Create DataLoaders for batch processing
            # shuffle=True on train — prevent model from learning order
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            return train_loader, test_loader
        """