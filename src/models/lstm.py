import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class ASLClassifier(nn.Module):
    """
    LSTM classifier for ASL landmark sequences.
    """

    def __init__(
        self,
        input_size: int = 126,
        hidden_size: int = 128,
        num_classes: int = 26,
    ):
        super().__init__()

        # Processes the sequence of video frames.
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
        )

        # Converts the final hidden state into class scores.
        self.fc = nn.Linear(
            hidden_size,
            num_classes,
        )

    def forward(self, videos, lengths):
        """
        videos shape:
            (batch_size, frames, 126)

        lengths:
            Original frame count for each video.
        """

        # Ignore zero-padding when processing the sequences.
        packed_videos = pack_padded_sequence(
            videos,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        # Get the final hidden state from the LSTM.
        _, (hidden, _) = self.lstm(packed_videos)

        # Use the final hidden state to predict the ASL class.
        output = self.fc(hidden[-1])

        return output