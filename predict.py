import torch
import typer
from pathlib import Path
from enum import Enum
import torchio as tio


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# certain volume of fmri sequence
timepoint = 0


# desired model
class ModelChoices(str, Enum):
    UNET2D = 'UNET2D'
    UNET3D = 'UNET3D'
    Red3D = 'REDUNET3D'
    VNET = 'VNET'


def main(input_dir: Path, output_dir: Path, sequence: bool, model_str: ModelChoices = typer.Argument('UNET2D')):
    # image files to predict the mask for
    if input_dir.is_file():
        image_paths = [input_dir]
    else:
        image_paths = list(input_dir.rglob('*.nii.gz'))

    # select model
    if model_str == ModelChoices.UNET2D:
        model = torch.load('output/k_cross.pth')
        shape = (160, 64, 35)
        transform = tio.CropOrPad(shape)
    elif model_str == ModelChoices.UNET3D:
        model = torch.load('output/unet3d.pth')
        shape = (160, 64, 48)
        transform = tio.CropOrPad(shape)
    elif model_str == ModelChoices.Red3D:
        model = torch.load('output/red3d.pth')
        shape = (160, 64, 48)
        transform = tio.CropOrPad(shape)
    elif model_str == ModelChoices.VNET:
        model = torch.load('output/vnet3d.pth')
        shape = (144, 48, 32)
        transform = tio.CropOrPad(shape)

    # save list with corresponding files and predicted masks
    with open('output/predictions/assignments.txt','w') as f:
        f.write(''.join(f'{img}\t{idx}\n' for idx, img in enumerate(image_paths)))

    # TODO: possible to predict for every timepoint in the case of sequence
    # TODO: create directory for output if not existent
    # make predictions
    for idx, img_path in enumerate(image_paths):
        img = tio.ScalarImage(img_path)
        # check if fmri sequence or mean was provided
        img = transform(img.data[timepoint][None, :]) if sequence else transform(img.data)
        img = img.reshape(1, 160, -1).to(device).unsqueeze(0)
        mask_data = model.predict(img)
        mask_data = (mask_data.squeeze().cpu().numpy().round()).reshape(shape)

        # get all (meta) information from image and save mask
        mask = tio.ScalarImage(img_path)
        mask.set_data(mask_data[None, :])
        original_shape = tio.ScalarImage(img_path).shape[1:]
        reverse_trans = tio.CropOrPad(original_shape)
        mask = reverse_trans(mask)
        mask.save(f'{output_dir}/predictions/pred_mask_{idx}.nii.gz')

    typer.echo(f'The predictions were made with {model_str}. The created masks are found in the output/predictions'
               f' folder assignments.txt provides the mapping from images to the indices of the mask files.')


if __name__ == '__main__':
    typer.run(main)
