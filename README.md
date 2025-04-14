# medical-segmentation
CV&amp;DL project NTNU 2025

UNet, TransUNet

From MONAI:
- Auto3DSeg
- AttentionUNet? [here](https://docs.monai.io/en/stable/networks.html#attentionunet), [paper](https://arxiv.org/pdf/1804.03999)

### History
## 14/04
- looked into the dataset
  - folder structure: top/patient/preRT/ MRI volume + tumor segmentation mask
  - GTV = gross tumor volume
- maybe use data augmentation? if training data is not enough
- [Auto3DSeg tutorial](https://github.com/Project-MONAI/tutorials/blob/main/auto3dseg/README.md)
