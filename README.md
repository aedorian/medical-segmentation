# medical-segmentation
CV&amp;DL project NTNU 2025

UNet, TransUNet

From MONAI:
- Auto3DSeg
- AttentionUNet? [here](https://docs.monai.io/en/stable/networks.html#attentionunet), [paper](https://arxiv.org/pdf/1804.03999)

### History
## 14/04
- looked into the dataset [link](https://zenodo.org/records/11199559)
  - folder structure: top/patient/preRT/ MRI volume + tumor segmentation mask
  - GTV = gross tumor volume
- maybe use data augmentation? if training data is not enough
- [Auto3DSeg tutorial](https://github.com/Project-MONAI/tutorials/blob/main/auto3dseg/README.md)

## 15/04
- no transunet in MONAI, but UNETR, swinunetr, dynamicunet are available
  - swin has better locality modeling than vit (closest to transunet)
  - dynunet has good performance and speed
- should use auto3dseg?
  - auto model selection: picks between unetr, swinunetr, dynunet based on data stats
  - auto transforms
  - but less control, heavy and harder experiment tracking
- [tutorial?](https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/swin_unetr_brats21_segmentation_3d.ipynb)
- [tutorial 2?](https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/swin_unetr_btcv_segmentation_3d.ipynb)
- [SwinUNETR paper](https://arxiv.org/pdf/2201.01266)
- final segmentation output: one channel for each class
- dice loss [link](https://cvinvolution.medium.com/dice-loss-in-medical-image-segmentation-d0e476eb486)

TO DO: fine tune model, see result of segmentation (viz)




1. Data Augmentation
Adding more data augmentation techniques can improve the model's generalization and robustness.

You can add rotations, flips, random zooms, and shifts. These will help the model handle different orientations and positions of the tumors.

Example:

python
Copy
Edit
train_transform = Compose([
    LoadImaged(keys=["image", "mask"]),
    EnsureChannelFirstd(keys=["image", "mask"]),
    Orientationd(keys=["image", "mask"], axcodes="RAS"),
    Spacingd(keys=["image", "mask"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
    ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True),
    CropForegroundd(keys=["image", "mask"], source_key="image"),
    ResizeWithPadOrCropd(keys=["image", "mask"], spatial_size=(64, 64, 64)),

    # Add augmentations
    RandRotate90d(keys=["image", "mask"], prob=0.5, max_k=3),  # Random rotation
    RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=[0, 1, 2]),  # Random flip in all axes
    RandShiftIntensityd(keys=["image"], prob=0.5, offset=0.1),  # Random intensity shift
    RandZoomd(keys=["image", "mask"], prob=0.5, min_zoom=0.9, max_zoom=1.1),  # Random zoom

    # Ensure mask is binary (0 or 1)
    Lambdad(keys=["mask"], func=lambda x: torch.clamp(torch.tensor(x), max=1)),

    EnsureTyped(keys=["image"], dtype=torch.float32),
    EnsureTyped(keys=["mask"], dtype=torch.long),

    ToTensord(keys=["image", "mask"]),
])
2. Use a Different Loss Function
Dice Loss is already a good choice for segmentation tasks, but adding a weighted loss can make the model pay more attention to harder-to-predict regions (like tumors, in your case).

You could try combining Dice Loss with Cross-Entropy Loss or using Dice and BCE (Binary Cross-Entropy) loss together.

Example:

python
Copy
Edit
from torch import nn

class DiceBCELoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceBCELoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.smooth = smooth

    def forward(self, outputs, targets):
        # Dice score
        intersection = torch.sum(outputs * targets)
        dice = (2. * intersection + self.smooth) / (torch.sum(outputs) + torch.sum(targets) + self.smooth)
        dice_loss = 1 - dice

        # BCE loss
        bce_loss = self.bce_loss(outputs, targets)

        return dice_loss + bce_loss

# Use this custom loss function
loss_function = DiceBCELoss()
3. Model Tuning
Increase feature size: Increasing the feature_size in SwinUNETR from 48 to a higher value (e.g., 64 or 128) may help improve performance, especially if you have enough GPU memory.

Learning rate adjustments: Sometimes the learning rate is too high or too low. You can try adjusting the learning rate scheduler, or you could experiment with learning rate warm-up or decay strategies.

Gradual unfreezing: If you’re using pre-trained weights, you can try gradually unfreezing layers to fine-tune the model more effectively.

4. Metrics & Validation
Use other metrics to track progress, like the IoU (Intersection over Union) or Sensitivity/Specificity.

Cross-validation: If you haven’t already, you might want to consider cross-validation to get a more stable estimate of the performance.

Visualize results: Visualizing the results with a few slices of the input and predicted outputs can help you see where the model is failing and might provide insights into improving the model.

5. Training for More Epochs
You’re currently training for 100 epochs. You might want to train for more epochs or use early stopping based on validation performance. Sometimes, models continue to improve well after 100 epochs.

6. Check for Data Imbalance
If one tumor type is underrepresented or the dataset is imbalanced, the model might have difficulty learning. Consider using class weights in your loss function or oversampling/undersampling the data.
