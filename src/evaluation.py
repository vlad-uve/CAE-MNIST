
import torch

def get_image_samples(validation_dataloader):

  # get a batch of images and labels from the validation set
  images, labels = next(iter(validation_dataloader))

  # select exactly one example for each digit (0-9)
  sample_labels = [i for i in range(10)]
  sample_indices = [torch.where(labels == i)[0][0].item() for i in range(10)]

  sample_images = images[sample_indices]

  return sample_images, sample_labels



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
