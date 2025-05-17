
import matplotlib.pyplot as plt
import torch

def plot_baseline_history(baseline_loss, to_plot_train=False):
    '''
    Plot loss history for the baseline model.

    Args:
        baseline_loss (dict): dictionary with 'epoch', 'train', and 'validation' lists
        to_plot_train (bool): if True, also plot training loss
    '''

    color=plt.get_cmap('tab10').colors

    # optionally plot training losses
    if to_plot_train:
        plt.plot(baseline_loss['epoch'], baseline_loss['train'], label='Base model (training loss)', color=color[0], linestyle='--')

    # plot validation losses
    plt.plot(baseline_loss['epoch'], baseline_loss['validation'], label='Base model (validation loss)', color=color[0], linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Baseline Model Loss')
    plt.legend()
    

def plot_experiment_history(loss_list, label_list, title, to_plot_train=False):
    '''
    Plot loss curves for multiple models.

    Args:
        loss_list (list of dict): list of loss history dictionaries (per model)
        label_list (list of str): list of model names (same length as loss_list)
        title (str): title for the plot
        to_plot_train (bool): if True, also plot training loss curves

    Each dictionary in loss_list must contain:
        - 'epoch': list of epoch numbers
        - 'train': list of training losses (optional if to_plot_train=False)
        - 'validation': list of validation losses
    '''

    color=plt.get_cmap('tab10').colors

    # loop over each loss history in the list
    for i, (loss_history, label) in enumerate(zip(loss_list, label_list)):
        # optionally plot training losses
        if to_plot_train:
            plt.plot(loss_history['epoch'], loss_history['train'], label=label + ' (training loss)', color=color[i+1], linestyle='--')

        # plot validation losses
        plt.plot(loss_history['epoch'], loss_history['validation'], label=label + ' (validation loss)', color=color[i+1], linewidth=2)

    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    

def plot_digits_row(images, labels=None, title=None, cmap='magma', figsize=(15, 3)):
    '''
    Display a row of digit images side by side.

    Args:
        images (numpy array or torch tensor): array of images (n_images, height, width)
        labels (list or array, optional): optional list of labels to display as titles
        title (str, optional): overall title for the plot
        cmap (str): matplotlib colormap for image display
        figsize (tuple): figure size for the plot
    '''

    n_images = images.shape[0]

    fig, axes = plt.subplots(1, n_images, figsize=figsize)

    for idx, ax in enumerate(axes.flat):
        # display each image
        ax.imshow(images[idx], cmap=cmap)
        ax.axis('off')

        # optionally set image label as title
        if labels is not None:
            ax.set_title(str(labels[idx]), fontsize=20)

    # optionally set a main title for the plot
    if title is not None:
        plt.suptitle(title, y=1, fontsize=30)

    plt.tight_layout()
    plt.show()

    # separator
    print('\n ')
    

def get_experiment_reconstructions(model_list, original_images, device):
    '''
    Run models on input images and return reconstructed outputs.

    Args:
        model_list (list): list of trained models to evaluate
        original_images (torch.Tensor): batch of original input images

    Returns:
        list of torch.Tensor: reconstructed images for each model
    '''
    reconstructions = []
    for model in model_list:
        model.eval()
        with torch.no_grad():
            reconstructed_images, _ = model(original_images.to(device))
            reconstructions.append(reconstructed_images.cpu())
    return reconstructions


def plot_experiment_reconstructions(reconstructions, labels, title_list):
    '''
    Plot reconstruction results for multiple models.

    Args:
        reconstructions (list of torch.Tensor): reconstructed outputs from models
        labels (list or array): labels for each image
        title_list (list of str): titles to display for each model
    '''
    for recon, title in zip(reconstructions, title_list):
        plot_digits_row(
            recon.squeeze(),
            labels,
            title=title + ' reconstructed digits'
        )
