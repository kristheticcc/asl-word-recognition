from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


class ASLDataset(Dataset):
    """
    Loads landmark sequences and their corresponding ASL word labels.
    """

    def __init__(
        self,
        metadata: pd.DataFrame,
        landmark_dir: Path,
        label_to_id: dict[str, int],
    ):
        # Reset indexes so dataset indexes start from zero.
        self.metadata = metadata.reset_index(drop=True)

        # Folder containing the .npy landmark files.
        self.landmark_dir = Path(landmark_dir)

        # Mapping from ASL words to numeric labels.
        self.label_to_id = label_to_id

    def __len__(self):
        """Return the number of videos in this dataset split."""
        return len(self.metadata)

    def __getitem__(self, index):
        """Load one video and return its features and label."""

        # Get the metadata row for this video.
        row = self.metadata.iloc[index]

        # WLASL video IDs are stored as five-digit filenames.
        video_id = int(row["video_id"])
        landmark_path = self.landmark_dir / f"{video_id:05d}.npy"

        # Stop with a clear error if the landmark file is missing.
        if not landmark_path.exists():
            raise FileNotFoundError(
                f"Landmark file not found: {landmark_path}"
            )

        # Load the landmark sequence.
        # Original shape: (frames, 2, 21, 3)
        landmarks = np.load(landmark_path)

        # Flatten each frame:
        # 2 hands × 21 landmarks × 3 coordinates = 126 features.
        features = landmarks.reshape(
            landmarks.shape[0],
            -1,
        )

        # Convert the ASL word into its numeric class ID.
        gloss = row["gloss"]
        label = self.label_to_id[gloss]

        # Return tensors for PyTorch.
        return (
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long),
        )


def collate_fn(batch):
    """
    Combines videos of different lengths into one padded batch.
    """

    # Separate video sequences from their labels.
    sequences, labels = zip(*batch)

    # Save the original frame count of every video.
    lengths = torch.tensor(
        [len(sequence) for sequence in sequences],
        dtype=torch.long,
    )

    # Pad shorter sequences with zeros.
    # Result shape:
    # (batch_size, longest_sequence, 126)
    padded_sequences = pad_sequence(
        list(sequences),
        batch_first=True,
        padding_value=0.0,
    )

    # Combine individual labels into one tensor.
    labels = torch.stack(list(labels))

    return padded_sequences, lengths, labels


def create_dataloaders(
    metadata_path: Path,
    landmark_dir: Path,
    batch_size: int = 16,
):
    """
    Create train, validation, and test DataLoaders.
    """

    # Load the processed metadata.
    metadata = pd.read_csv(metadata_path)

    # Create a consistent alphabetically sorted label mapping.
    labels = sorted(metadata["gloss"].unique())

    label_to_id = {
        label: index
        for index, label in enumerate(labels)
    }

    id_to_label = {
        index: label
        for label, index in label_to_id.items()
    }

    # Keep the official WLASL splits.
    train_metadata = metadata[
        metadata["split"] == "train"
    ]

    val_metadata = metadata[
        metadata["split"] == "val"
    ]

    test_metadata = metadata[
        metadata["split"] == "test"
    ]

    # Create one dataset object for each split.
    train_dataset = ASLDataset(
        train_metadata,
        landmark_dir,
        label_to_id,
    )

    val_dataset = ASLDataset(
        val_metadata,
        landmark_dir,
        label_to_id,
    )

    test_dataset = ASLDataset(
        test_metadata,
        landmark_dir,
        label_to_id,
    )

    # Create DataLoaders.
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        label_to_id,
        id_to_label,
    )