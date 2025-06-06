
import os
import torch
import shutil
import subprocess

def save_experiment_files(
    experiment_name,
    models,
    losses,
    reconstructions,
    param_counts_dict,
    description_text,
    local_path_root='/content'
):
    """
    Save experiment files: model weights, loss history, reconstructions, and description.

    Args:
        experiment_name (str): e.g., "experiment_2"
        models (list): list of trained model objects
        losses (list): list of loss history objects
        reconstructions (list): list of reconstructed image tensors
        description_text (str): plain-text description
        local_path_root (str): where to create the export folder (default: '/content')
    """

    export_folder = os.path.join(local_path_root, f'CAE_{experiment_name}_local')
    os.makedirs(export_folder, exist_ok=True)

    for idx, (model, loss, recon) in enumerate(zip(models, losses, reconstructions)):
        torch.save(model.state_dict(), os.path.join(export_folder, f'{experiment_name}_model_{idx+1}.pth'))
        torch.save(loss, os.path.join(export_folder, f'{experiment_name}_loss_{idx+1}.pth'))
        torch.save(recon, os.path.join(export_folder, f'{experiment_name}_reconstruction_{idx+1}.pth'))
        
    torch.save(param_counts_dict, os.path.join(export_folder, f'{experiment_name}_param_counts.pth'))

    with open(os.path.join(export_folder, f'{experiment_name}_description.txt'), 'w') as f:
        f.write(description_text.strip())

    print(f"✅ Saved {experiment_name} files to: {export_folder}")


def export_experiment_files(experiment_name, model_count,
                            local_root='/content',
                            repo_root='/content/CAE-MNIST'):
    """
    Copies experiment output files from local folder to Git repo and pushes them.

    Args:
        experiment_name (str): e.g. "experiment_2"
        model_count (int): number of models/files to export
        local_root (str): path where local files are stored (default: /content)
        repo_root (str): path to the cloned Git repo (default: /content/CAE-MNIST)
    """

    # Define folders
    local_export_folder = os.path.join(local_root, f'CAE_{experiment_name}_local')
    git_output_folder = os.path.join(repo_root, 'outputs', f'{experiment_name}_files')
    os.makedirs(git_output_folder, exist_ok=True)

    # Gather file names
    files_to_copy = []

    # Collect filenames to copy
    for idx in range(0, model_count):
      files_to_copy.append(f'{experiment_name}_model_{idx+1}.pth')
      files_to_copy.append(f'{experiment_name}_loss_{idx+1}.pth')
      files_to_copy.append(f'{experiment_name}_reconstruction_{idx+1}.pth')

    # Model parameters
    files_to_copy.append(f'{experiment_name}_param_counts.pth')

    # Add description
    files_to_copy.append(f'{experiment_name}_description.txt')

    # Copy files into Git folder
    for file in files_to_copy:
        shutil.copy2(
            os.path.join(local_export_folder, file),
            os.path.join(git_output_folder, file)
        )

    # Git add, commit, push
    os.chdir(repo_root)
    os.system(f'git add outputs/{experiment_name}_files/*')
    os.system(f'git commit -m "Add {experiment_name}: models, losses, reconstructions, and description" || echo "Nothing to commit"')
    os.system('git push origin main')

    print(f"✅ Exported {experiment_name} files to: outputs/{experiment_name}_files/")
