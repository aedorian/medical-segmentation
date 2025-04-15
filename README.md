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
