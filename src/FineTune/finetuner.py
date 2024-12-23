import torch
import os.path as osp
from pathlib import Path
from os import environ as env


def printProgressBar(
    iteration,
    total,
    prefix="",
    suffix="",
    decimals=1,
    length=100,
    fill="█",
    printEnd="\r",
):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + "-" * (length - filledLength)
    print(f"\r{prefix} |{bar}| {percent}% {suffix}", end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


def write_to_log(save_path: Path, text: str) -> None:
    """Writes text to the log file

    Args:
        save_path (Path): The save path
        text (str): The text to write
    """
    with open(osp.join(save_path, "log.txt"), "a") as f:
        f.write(text + "\n")


class FineTuner:
    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        dataset_name,
        train_dataloader,
        val_dataloader,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device)
        self.dataset_name = dataset_name
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler

    def train_step(self, batch: tuple[torch.tensor, torch.tensor]) -> float:
        """A single training step

        Args:
            batch (tuple[torch.tensor, torch.tensor]): The batch

        Returns:
            float: The loss
        """
        # Prepare the inputs
        t_input_ids = batch[0].to(self.device)
        t_labels = batch[0].to(
            self.device
        )  # Labels for GPT-2 are usually the same as inputs
        t_attn_mask = batch[1].to(self.device)

        # Clear gradients from the previous step
        self.optimizer.zero_grad()

        # Forward pass
        t_outputs = self.model(
            input_ids=t_input_ids,
            labels=t_labels,
            attention_mask=t_attn_mask,
        )

        # Compute the loss
        t_loss = t_outputs.loss

        # Backpropagate the gradients
        t_loss.backward()

        # Step the optimizer
        self.optimizer.step()

        # Step the learning rate scheduler (if applicable)
        self.scheduler.step()

        # Return the loss for monitoring
        return t_loss.item()

    def get_avg_val_loss(self) -> int:
        """Get the average validation loss

        Returns:
            int: The average validation loss
        """
        self.model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in self.val_dataloader:
                v_input_ids = batch[0].to(self.device)  # Input IDs (text data)
                v_labels = batch[0].to(self.device)  # Labels are the same as input_ids
                v_attn_mask = batch[1].to(self.device)  # Attention mask
                v_outputs = self.model(
                    input_ids=v_input_ids,
                    labels=v_labels,
                    attention_mask=v_attn_mask,
                )
                v_loss = v_outputs.loss
                total_val_loss += v_loss.item()
        return total_val_loss / len(self.val_dataloader)

    def get_save_path(self) -> Path:
        """Get the save path

        Returns:
            str: The save path
        """
        return Path(osp.join(env["selfChatBot_results"], self.dataset_name))

    def fine_tune(self, epochs: int):
        """Fine tune the model

        Args:
            epochs (int): The number of epochs
        """
        train_size = len(self.train_dataloader)
        save_path = self.get_save_path()
        best_loss = 1e9
        save_path.mkdir(parents=True, exist_ok=True)
        open(osp.join(save_path, "log.txt"), "w").close()
        for epoch in range(epochs):
            self.model.train()
            total_train_loss = 0.0
            for step, batch in enumerate(self.train_dataloader):
                printProgressBar(
                    step + 1,
                    train_size,
                    prefix=f"Epoch: {
                        epoch + 1}",
                    suffix=f"{step + 1}/{train_size}",
                )
                loss = self.train_step(batch)
                total_train_loss += loss
            val_loss = self.get_avg_val_loss()
            result_string = f"Epoch: {epoch + 1}, Train Loss: {total_train_loss / train_size}, Validation Loss: {val_loss}"
            print(result_string)
            if val_loss < best_loss:
                best_loss = val_loss
                print(f"New best validation loss: {best_loss}")
                result_string += f", New best validation loss: {best_loss}"
                self.model.save_pretrained(save_path)
            write_to_log(save_path, result_string)
