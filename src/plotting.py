
import matplotlib.pyplot as plt
import torch
from IPython.display import Image, display

def plot_experiment_history(loss_list, label_list, title, to_plot_train=False, color=plt.get_cmap('tab10').colors):
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

    # loop over each loss history in the list
    for i, (loss_history, label) in enumerate(zip(loss_list, label_list)):

      # optionally plot training losses
      if to_plot_train:
          plt.plot(loss_history['epoch'], loss_history['train'], label=label + ' (train)', color=color[i], linestyle='--')

      # plot validation losses
      plt.plot(loss_history['epoch'], loss_history['validation'], label=label + ' (val)', color=color[i], linewidth=2)

    plt.xlabel('Epoch')
    plt.ylabel('Loss')


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

    return fig


def plot_experiment_reconstructions(reconstructions, labels, title_list):
    '''
    Plot reconstruction results for multiple models.

    Args:
        reconstructions (list of torch.Tensor): reconstructed outputs from models
        labels (list or array): labels for each image
        title_list (list of str): titles to display for each model
    '''

    figures = []

    for recon, title in zip(reconstructions, title_list):
        fig = plot_digits_row(
            recon.squeeze(),
            labels,
            title=title + ' reconstruction'
        )

        figures.append(fig)

    return figures


def display_reconstruction_images(experiment_number, model_count):
    """
    Display base model reconstruction and experiment reconstruction images.

    Args:
        experiment_number (int): experiment number (e.g., 2 for 'experiment_2').
        model_count (int): number of experiment models to display.
    """
    # display baseline reconstruction
    display(Image(filename='/content/CAE-MNIST/outputs/base_model_files/base_image_reconstruction.png'))

    # display experiment reconstructions
    for idx in range(1, model_count + 1):
        path = f'/content/CAE-MNIST/outputs/experiment_{experiment_number}_files/experiment_{experiment_number}_image_reconstruction_{idx}.png'
        display(Image(filename=path))
