import torch
import torch.nn as nn


def train_one_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
):
    """
    Train the model for one epoch.
    """

    model.train()
    total_loss = 0.0

    for videos, lengths, labels in train_loader:
        videos = videos.to(device)
        labels = labels.to(device)

        # Clear gradients from the previous batch.
        optimizer.zero_grad()

        # Make predictions.
        outputs = model(videos, lengths)

        # Compare predictions with the correct labels.
        loss = criterion(outputs, labels)

        # Calculate gradients and update model weights.
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Return the average loss for this epoch.
    return total_loss / len(train_loader)


def train_model(
    model,
    train_loader,
    num_epochs=20,
    learning_rate=0.001,
):
    """
    Train the LSTM model for multiple epochs.
    """

    # Use Apple's MPS device when available.
    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    model = model.to(device)

    # CrossEntropyLoss is used for 26-class classification.
    criterion = nn.CrossEntropyLoss()

    # Adam updates the model parameters.
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    training_losses = []

    for epoch in range(num_epochs):
        average_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        training_losses.append(average_loss)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"Loss: {average_loss:.4f}"
        )

    return model, training_losses, device