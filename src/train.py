
import torch
import torch.nn.functional as F

def train_model(model, train_dataloader, optimizer, epoch, device):
    """
    Runs one training epoch for the given model.

    Args:
        model (nn.Module): the autoencoder to train
        train_dataloader (DataLoader): training data loader
        optimizer (torch.optim.Optimizer): optimizer used for training
        epoch (int): current epoch number (used for tracking/logging)

    Returns:
        float: loss value from the last batch of the epoch
    """

    # set model to training mode
    model.train()

    for b_i, (input_x, _) in enumerate(train_dataloader):
        # move batch to device
        input_x = input_x.to(device)

        # clear previous gradients
        optimizer.zero_grad()

        # forward pass: get model output
        decoded_x, encoded_x = model(input_x)

        # compute reconstruction loss between input and output
        loss = F.binary_cross_entropy(decoded_x, input_x)

        # backward pass: compute gradients
        loss.backward()

        # update weights
        optimizer.step()

    # return last batch loss
    return loss.item()


def validate_model(model, validation_dataloader, device):
    """
    Evaluates the model on the validation set using binary cross-entropy loss.

    Args:
        model (nn.Module): trained autoencoder
        validation_dataloader (DataLoader): validation data loader
        device (str): 'cuda' or 'cpu'

    Returns:
        float: average loss over the entire validation set
    """

    # set model to evaluation mode (disables dropout, batchnorm updates etc.)
    model.eval()
    total_loss = 0

    # disable gradient calculation
    with torch.no_grad():
        for input_x, _ in validation_dataloader:
            # move batch to device
            input_x = input_x.to(device)

            # forward pass
            decoded_x, encoded_x = model(input_x)

            # accumulate reconstruction loss for each bath
            total_loss += F.binary_cross_entropy(decoded_x, input_x)

    # compute and return average loss over validation over one epoch
    avg_loss = total_loss / len(validation_dataloader)

    return avg_loss.item()


def run_model_training(model, train_dataloader, validation_dataloader, optimizer, scheduler, num_epoch, device):
    """
    Trains the model across multiple epochs and evaluates on validation set.

    Args:
        model (nn.Module): the autoencoder
        train_dataloader (DataLoader): training data
        validation_dataloader (DataLoader): validation data
        optimizer (Optimizer): optimizer for training
        scheduler (LRScheduler): learning rate scheduler
        num_epoch (int): number of training epochs

    Returns:
        model: trained model
        dict: loss history containing 'train', 'validation', and 'epoch' lists
    """

    # initialize loss tracking dictionary
    loss_history = {'train': [], 'validation': [], 'epoch': []}

    print('\nTRAINING IS STARTED:')

    # run training loop
    for epoch in range(1, num_epoch + 1):
        # train model on training set
        train_loss = train_model(model, train_dataloader, optimizer, epoch, device)

        # evaluate model on validation set
        validation_loss = validate_model(model, validation_dataloader)

        # check if scheduler reduces learning rate based on validation loss plateau
        previous_lr = optimizer.param_groups[0]['lr']
        scheduler.step(validation_loss)
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr != previous_lr:
            print(f"LR reduced from {previous_lr:.4f} → {current_lr:.4f}")

        # record losses and epoch number
        loss_history['train'].append(train_loss)
        loss_history['validation'].append(validation_loss)
        loss_history['epoch'].append(epoch)

        # print progress
        print(f"Epoch {epoch:2d} | Train Loss: {train_loss:.4f} | Validation Loss: {validation_loss:.4f}")

    print('\nTRAINING IS FINISHED.')

    return model, loss_history
