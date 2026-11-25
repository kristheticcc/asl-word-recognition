from pathlib import Path

import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from .data.dataset import create_dataloaders
from .models.lstm import ASLClassifier

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def collect_predictions(model, data_loader, device):
    """
    Run the model on a dataset and collect predictions and labels.
    """

    model.eval()

    all_predictions = []
    all_labels = []

    # Disable gradient calculations during evaluation.
    with torch.no_grad():
        for videos, lengths, labels in data_loader:

            # Move video data to the selected device.
            videos = videos.to(device)

            # Keep lengths on the CPU because
            # pack_padded_sequence requires CPU lengths.
            outputs = model(videos, lengths)

            # Select the class with the highest score.
            predictions = outputs.argmax(dim=1).cpu()

            all_predictions.extend(predictions.tolist())
            all_labels.extend(labels.tolist())

    return all_labels, all_predictions


def main():
    # Find the project root directory.
    project_root = Path(__file__).resolve().parents[1]

    # Define the processed dataset paths.
    metadata_path = (
        project_root
        / "data"
        / "preprocessed"
        / "metadata.csv"
    )

    landmark_dir = (
        project_root
        / "data"
        / "preprocessed"
        / "landmarks"
    )

    # Define the saved baseline checkpoint path.
    checkpoint_path = (
        project_root
        / "results"
        / "checkpoints"
        / "baseline_lstm.pth"
    )

    # Select CPU or Apple MPS.
    device = get_device()
    print("Using device:", device)

    # Create the DataLoaders and label mappings.
    (
        train_loader,
        val_loader,
        test_loader,
        label_to_id,
        id_to_label,
    ) = create_dataloaders(
        metadata_path=metadata_path,
        landmark_dir=landmark_dir,
        batch_size=16,
    )

    # Load the saved checkpoint onto the CPU first.
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    # Get the model settings saved in the checkpoint.
    model_config = checkpoint["model_config"]

    # Recreate the same LSTM architecture.
    model = ASLClassifier(**model_config)

    # Load the trained weights.
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # Move the model to the selected device.
    model = model.to(device)

    # Make sure the current label mapping matches
    # the mapping used when the checkpoint was trained.
    saved_label_to_id = checkpoint.get("label_to_id")

    if saved_label_to_id != label_to_id:
        raise ValueError(
            "The current label mapping does not match "
            "the checkpoint label mapping."
        )

    # Evaluate on the validation set to match the notebook.
    all_labels, all_predictions = collect_predictions(
        model=model,
        data_loader=val_loader,
        device=device,
    )

    # Create class IDs and names in the correct order.
    class_ids = list(range(len(label_to_id)))

    class_names = [
        id_to_label[class_id]
        for class_id in class_ids
    ]

    # Calculate validation accuracy.
    validation_accuracy = accuracy_score(
        all_labels,
        all_predictions,
    )

    print(
        f"Validation accuracy: "
        f"{validation_accuracy * 100:.2f}%"
    )

    # Display the confusion matrix.
    confusion = confusion_matrix(
        all_labels,
        all_predictions,
        labels=class_ids,
    )

    print("\nConfusion matrix:")
    print(confusion)

    # Display precision, recall, and F1-score.
    print("\nClassification report:")
    print(
        classification_report(
            all_labels,
            all_predictions,
            labels=class_ids,
            target_names=class_names,
            zero_division=0,
        )
    )

    # Display the accuracy saved inside the checkpoint.
    saved_accuracy = checkpoint.get(
        "validation_accuracy"
    )

    if saved_accuracy is not None:
        print(
            "Checkpoint accuracy: "
            f"{saved_accuracy * 100:.2f}%"
        )


if __name__ == "__main__":
    main()