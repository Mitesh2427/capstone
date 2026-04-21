import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

SFREQ = 250
TMIN = 0.5
TMAX = 6.0


USE_RANDOM_SPLIT = True


# ─────────────────────────────
# LOAD DATA
# ─────────────────────────────
def load_dataset(subject_ids=None):
    from moabb.datasets import BNCI2014001

    dataset = BNCI2014001()

    if subject_ids is None:
        subject_ids = dataset.subject_list

    return dataset, subject_ids


# ─────────────────────────────
# PARADIGM (OPTIMAL BAND)
# ─────────────────────────────
def get_paradigm():
    from moabb.paradigms import MotorImagery

    return MotorImagery(
        events=["left_hand", "right_hand", "feet", "tongue"],
        n_classes=4,
        tmin=TMIN,
        tmax=TMAX,
        resample=SFREQ,
        fmin=8,
        fmax=30,
    )


# ─────────────────────────────
# DATA EXTRACTION (FIXED)
# ─────────────────────────────
def get_subject_data(dataset, paradigm, subject_id):
    epochs, labels, meta = paradigm.get_data(
        dataset=dataset,
        subjects=[subject_id],
        return_epochs=True
    )

    X = epochs.get_data()
    ch_names = epochs.ch_names

    
    if USE_RANDOM_SPLIT:
        X_train, X_test, y_train, y_test = train_test_split(
            X, labels,
            test_size=0.2,
            stratify=labels,
            random_state=42
        )

    
    else:
        sessions = meta["session"].values
        unique_sessions = np.unique(sessions)

        train_mask = sessions == unique_sessions[0]
        test_mask = sessions == unique_sessions[1]

        X_train, y_train = X[train_mask], labels[train_mask]
        X_test, y_test = X[test_mask], labels[test_mask]

    return {
        "train": {"X": X_train, "y": y_train},
        "test": {"X": X_test, "y": y_test},
        "ch_names": ch_names
    }


# ─────────────────────────────
# CHANNEL CLEANING
# ─────────────────────────────
def select_eeg_channels(X, ch_names):
    eeg_idx = [i for i, ch in enumerate(ch_names) if "EOG" not in ch.upper()]
    return X[:, eeg_idx, :], [ch_names[i] for i in eeg_idx]


# ─────────────────────────────
# LABEL ENCODING
# ─────────────────────────────
def encode_labels(y_train, y_test):
    le = LabelEncoder()
    le.fit(y_train)

    return (
        le.transform(y_train),
        le.transform(y_test),
        dict(zip(le.classes_, range(len(le.classes_))))
    )


# ─────────────────────────────
#  STRONG NORMALIZATION
# ─────────────────────────────
def normalize(X_train, X_test):

    def trial_norm(X):
        mean = X.mean(axis=2, keepdims=True)
        std = X.std(axis=2, keepdims=True)

        std[std < 1e-6] = 1e-6
        X = (X - mean) / std

        
        X = np.clip(X, -5, 5)

        return X

    X_train = trial_norm(X_train)
    X_test = trial_norm(X_test)

    return X_train, X_test


# ─────────────────────────────
# FORMAT FOR EEGNET
# ─────────────────────────────
def format_eegnet(X):
    return X[:, np.newaxis, :, :].astype(np.float32)


# ─────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────
def preprocess_subject(dataset, paradigm, subject_id):
    data = get_subject_data(dataset, paradigm, subject_id)

    X_train, ch_names = select_eeg_channels(
        data["train"]["X"], data["ch_names"]
    )
    X_test, _ = select_eeg_channels(
        data["test"]["X"], data["ch_names"]
    )

    y_train, y_test, label_map = encode_labels(
        data["train"]["y"],
        data["test"]["y"]
    )

    X_train, X_test = normalize(X_train, X_test)

    return {
        "X_train": format_eegnet(X_train),
        "y_train": y_train.astype(np.int64),
        "X_test": format_eegnet(X_test),
        "y_test": y_test.astype(np.int64),
        "label_map": label_map,
    }


# ─────────────────────────────
# RUN ALL SUBJECTS
# ─────────────────────────────
def run_pipeline(subject_ids=None):
    dataset, subject_ids = load_dataset(subject_ids)
    paradigm = get_paradigm()

    data = {}
    for subj in subject_ids:
        data[subj] = preprocess_subject(dataset, paradigm, subj)

    return data