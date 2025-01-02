import torch
from pathlib import Path
from torch.cuda.amp import autocast, GradScaler
from contextlib import nullcontext


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


def get_best_and_final_path(save_path: Path) -> tuple[Path, Path]:
    """Returns the best and final path

    Args:
        save_path (Path): The save path

    Returns:
        tuple[Path, Path]: The best and final path
    """
    best_path, final_path = save_path.joinpath("best"), save_path.joinpath("final")
    best_path.mkdir(exist_ok=True)
    final_path.mkdir(exist_ok=True)
    return best_path, final_path


def write_to_log(save_path: Path, text: str) -> None:
    """Writes text to the log file

    Args:
        save_path (Path): The save path
        text (str): The text to write
    """
    with open(save_path.joinpath("log.txt"), "a") as f:
        f.write(text + "\n")


class FineTuner:

    PATIENCE: int = 8

    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        save_path,
        train_dataloader,
        val_dataloader,
        fp16=False,
        gradient_accumulation_steps=1,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device)
        self.save_path = save_path
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.patience_counter = 0
        self.fp16 = fp16
        self.gradient_accumulation_steps = gradient_accumulation_steps
        if self.fp16:
            self.scaler = GradScaler(self.device)

    def train_step(self, step: int, batch: tuple[torch.tensor, torch.tensor]) -> float:
        """A single training step

        Args:
            batch (tuple[torch.tensor, torch.tensor]): The batch

        Returns:
            float: The loss
        """
        # Prepare the inputs
        t_input_ids = batch[0].to(self.device)
        t_labels = batch[0].to(self.device)
        t_attn_mask = batch[1].to(self.device)

        with autocast(self.device) if self.fp16 else nullcontext():
            # Forward pass
            t_outputs = self.model(
                input_ids=t_input_ids,
                labels=t_labels,
                attention_mask=t_attn_mask,
            )
            # Compute the loss
            t_loss = t_outputs.loss
            # Divide the loss by the gradient accumulation steps to account for the accumulated gradients
            t_loss /= self.gradient_accumulation_steps

        if self.fp16:
            # Scale the loss for backpropagation if using fp16
            self.scaler.scale(t_loss).backward()
        else:
            # Backpropagate the gradients
            t_loss.backward()

        if (step + 1) % self.gradient_accumulation_steps == 0:
            if self.fp16:
                # Step the optimizer
                self.scaler.step(self.optimizer)
                # Update the scaler
                self.scaler.update()
            else:
                # Step the optimizer
                self.optimizer.step()
            # Step the learning rate scheduler
            self.scheduler.step()
            # Zero the gradients
            self.optimizer.zero_grad()

        # Return the loss for monitoring
        return t_loss.item()

    def gradient_accumulation_final_update(self, step: int):
        """Final update for gradient accumulation

        Args:
            step (int): The current step number
        """
        if (step + 1) % self.gradient_accumulation_steps != 0:
            if self.fp16:
                # Update model weights with accumulated gradients
                self.scaler.step(self.optimizer)
                # Update the scaler
                self.scaler.update()
            else:
                self.optimizer.step()
            # Step the learning rate scheduler
            self.scheduler.step()
            # Zero the gradients
            self.optimizer.zero_grad()

    def train_epoch(self, epoch: int) -> float:
        """Perform a single epoch of training

        Args:
            epoch (int): The current epoch number

        Returns:
            float: The average training loss
        """
        total_train_loss = 0.0
        train_size = len(self.train_dataloader)
        for step, batch in enumerate(self.train_dataloader):
            printProgressBar(
                step + 1,
                train_size,
                prefix=f"Epoch: {
                    epoch + 1}",
                suffix=f"{step + 1}/{train_size}",
            )
            loss = self.train_step(step, batch)
            total_train_loss += loss
        self.gradient_accumulation_final_update(step)
        return total_train_loss / train_size

    def get_avg_val_loss(self) -> float:
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

    def fine_tune(self):
        """Fine tune the model

        Args:
            epochs (int): The number of epochs
        """
        best_loss = 1e9
        best_path, final_path = get_best_and_final_path(self.save_path)
        open(self.save_path.joinpath("log.txt"), "w").close()
        for epoch in range(self.parameters["epochs"]):
            self.model.train()
            train_loss = self.train_epoch(epoch)
            val_loss = self.get_avg_val_loss()
            result_string = f"Epoch: {epoch + 1}, Train Loss: {train_loss}, Validation Loss: {val_loss}"
            print(result_string)
            if val_loss < best_loss:
                best_loss = val_loss
                self.patience_counter = 0
                print(f"New best validation loss: {best_loss}")
                result_string += f", New best validation loss: {best_loss}"
                self.model.save_pretrained(best_path)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.PATIENCE:
                    print(f"Early stopping at epoch {epoch + 1}")
                    result_string += f", Early stopping at epoch {epoch + 1}"
                    self.model.save_pretrained(final_path)
            write_to_log(self.save_path, result_string)
            if self.patience_counter >= self.PATIENCE:
                break
