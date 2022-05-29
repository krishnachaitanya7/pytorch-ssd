import argparse
import glob
import re

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single Shot MultiBox Detector Training With PyTorch")
    parser.add_argument(
        "--checkpoint-folder", "--model-dir", default="models/", help="Directory for saving checkpoint models"
    )
    args = parser.parse_args()
    models_folder = args.checkpoint_folder
    # recursively find all the files ending with .pth in the folder
    files = glob.glob(models_folder + "/*.pth", recursive=True)
    # get the loss in the file of the format mb1-ssd-Epoch-XX-Loss-XX.pth
    loss_regex = re.compile(r"mb1-ssd-Epoch-\d+-Loss-(\d+\.\d+).pth")
    # get the epoch number in the file of the format mb1-ssd-Epoch-XX-Loss-XX.pth
    epoch_regex = re.compile(r"mb1-ssd-Epoch-(\d+)-Loss-\d+\.\d+.pth")
    # get the loss and epoch number from the file name
    losses = [float(loss_regex.search(f).group(1)) for f in files]
    epochs = [int(epoch_regex.search(f).group(1)) for f in files]
    # find the best model based on the lowest loss
    best_model_idx = losses.index(min(losses))
    print(
        f"Best model is epoch {epochs[best_model_idx]} with loss {losses[best_model_idx]}, with filename: {files[best_model_idx]}"
    )
