from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from plr_exercise.models.cnn import Net

import wandb
import optuna
from timing import Timer, CpuTimer  # Import the Timer and CpuTimer




"""class Net(nn.Module):
    def __init__(self):

        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output"""


def train(args, model, device, train_loader, optimizer, epoch):
    """
    Trains the model for one epoch.

    Args:
        args: Command-line arguments.
        model (Net): The neural network model to be trained.
        device (torch.device): The device to perform the training on (CPU or CUDA).
        train_loader (DataLoader): The DataLoader for training data.
        optimizer (torch.optim.Optimizer): The optimizer for updating model parameters.
        epoch (int): The current epoch number.

    Trains the model on the training dataset and logs the training progress.
    """

    model.train()
    # start a new wandb run to track this script
    """wandb.init(project="my-plr-ml-project",
               config={
                "learning_rate": args.lr,
                "architecture": "CNN",
                "dataset": "MNIST",
                "epochs": args.epochs,
                }       
    )"""

    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
            wandb.log({"training_loss:": loss.item()})
            if args.dry_run:
                break


def test(model, device, test_loader, epoch):
    """
    Evaluates the model on the test dataset.

    Args:
        model (Net): The trained neural network model.
        device (torch.device): The device to perform the evaluation on (CPU or CUDA).
        test_loader (DataLoader): The DataLoader for test data.
        epoch (int): The current epoch number.

    Returns:
        float: The accuracy of the model on the test dataset.
    
    Evaluates the model's performance on the test set and logs the results.
    """

    model.eval()
   
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:

            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100.0 * correct / len(test_loader.dataset)

    log_image_interval = 1
    if epoch % log_image_interval == 0:  # Log images every 'log_image_interval' epochs
        images = data.cpu().numpy()[:10]  # Log first 10 images of the batch
        preds = pred.squeeze().cpu().numpy()[:10]
        actuals = target.cpu().numpy()[:10]

        for i, (img, pred, actual) in enumerate(zip(images, preds, actuals)):
            wandb.log({"test_images": [wandb.Image(img, caption=f"Pred: {pred}, Actual: {actual}")]})


    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
        )
    )
    wandb.log({"testing_loss:": test_loss})

    return accuracy



def main():
    """
    The main function that sets up the training and testing environment, parses command-line arguments,
    and initiates the training and testing process.

    It initializes the model, data loaders, optimizer, and scheduler, and then starts the training process.
    It also handles hyperparameter optimization using Optuna.
    """

    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--batch-size", type=int, default=64, metavar="N", help="input batch size for training (default: 64)"
    )
    parser.add_argument(
        "--test-batch-size", type=int, default=1000, metavar="N", help="input batch size for testing (default: 1000)"
    )
    parser.add_argument("--epochs", type=int, default=2, metavar="N", help="number of epochs to train (default: 14)")
    parser.add_argument("--lr", type=float, default=1.0, metavar="LR", help="learning rate (default: 1.0)")
    parser.add_argument("--gamma", type=float, default=0.7, metavar="M", help="Learning rate step gamma (default: 0.7)")
    parser.add_argument("--no-cuda", action="store_true", default=False, help="disables CUDA training")
    parser.add_argument("--dry-run", action="store_true", default=False, help="quickly check a single pass")
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument("--save-model", action="store_true", default=False, help="For Saving the current Model")
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    DEVICE = device
    train_kwargs = {"batch_size": args.batch_size}
    test_kwargs = {"batch_size": args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset1 = datasets.MNIST("../data", train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST("../data", train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    
    


    def objective(trial):
        """
        Defines the objective function for hyperparameter optimization using Optuna.

        This function sets up the model, optimizer, and learning rate scheduler with hyperparameters
        suggested by Optuna. It then trains the model for a number of epochs and evaluates its
        performance on the test set. The function returns the accuracy of the model on the test set,
        which serves as the value to be optimized by Optuna.

        Args:
            trial (optuna.trial.Trial): An Optuna trial object which suggests hyperparameters.

        Returns:
            float: The accuracy of the model on the test dataset, which Optuna will attempt to maximize.

        The function trains the model using the specified hyperparameters and evaluates its performance,
        reporting the results back to Optuna. It also includes functionality to prune trials that do not
        meet certain criteria, enhancing the efficiency of the hyperparameter optimization process.
        """
        model = Net().to(device)

        lr = trial.suggest_float("lr", 1e-6, 1e-2)
        gamma = trial.suggest_float("gamma", 0.1, 0.9)
        epochs = trial.suggest_int("epochs", 1, 3)

        print("Training with parameters: --lr:", lr, "--gamma:",gamma,'--epochs:',epochs)

        wandb.init(project="plr_exercise",
                   config={
                       "learning_rate": lr,
                        "epochs": epochs,
                        "gamma": gamma
                        },
                    name=f'trial-{trial.number}',
                    )

        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

        for epoch in range(epochs):
            train(args, model, DEVICE, train_loader, optimizer, epoch)
            accuracy = test(model, DEVICE, test_loader, epoch)
            scheduler.step()
        
        trial.report(accuracy, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
        if args.save_model:
            torch.save(model.state_dict(), "mnist_cnn.pt")
       

        return accuracy

    study = optuna.create_study()
    study.optimize(objective, n_trials=3)

    print(study.best_params)
        
  

    

    wandb.finish()


if __name__ == "__main__":
    main()
