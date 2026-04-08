# Run Tracking Log

## 2026-04-07 11:56:37 | dataloader_check

### Summary
- split: train
- augment: True
- dataset_len: 1871
- batch_size: 2
- image_shape: (2, 1, 256, 256)
- mask_shape: (2, 1, 256, 256)
- hard_thin_unique: [0]
- hard_thin_is_binary: True

### Details
- metadata_csv: data\processed\metadata.csv
- processed_root: data\processed
- example_imt_mm: [0.652672, 0.518618]
- example_hard_thin: [0, 0]


## 2026-04-07 11:57:12 | preprocess

### Summary
- processed_total: 2
- failed_total: 0
- processed_cubs_2021: 1
- processed_cubs_2022: 1

### Details
- config_path: configs\config.yaml
- limit: 1
- metadata_csv: D:\Kuliah\S2ElektronikaITS\Tesis\Riset\GenerativeAI\data\processed\metadata.csv
- failed_csv: D:\Kuliah\S2ElektronikaITS\Tesis\Riset\GenerativeAI\data\processed\failed_samples.csv


## 2026-04-08 09:20:42 | segmentation_sanity_train

### Summary
- model: unet
- device: cpu
- epochs: 3
- train_samples: 2
- val_samples: 0
- batch_size: 2
- visuals_saved: 0
- best_val_dice: 0.0
- last_val_dice: 0.0

### Details
- output_root: outputs\seg_sanity\20260408_092041
- history_csv: outputs\seg_sanity\20260408_092041\history.csv
- checkpoint: outputs\seg_sanity\20260408_092041\checkpoint_last.pt
- visuals_dir: outputs\seg_sanity\20260408_092041\visuals


## 2026-04-08 09:22:22 | segmentation_sanity_train

### Summary
- model: unet
- device: cpu
- epochs: 3
- train_samples: 2
- val_samples: 0
- batch_size: 2
- visuals_saved: 0
- best_val_dice: 0.0
- last_val_dice: 0.0

### Details
- output_root: outputs\seg_sanity\20260408_092221
- history_csv: outputs\seg_sanity\20260408_092221\history.csv
- checkpoint: outputs\seg_sanity\20260408_092221\checkpoint_last.pt
- visuals_dir: outputs\seg_sanity\20260408_092221\visuals


## 2026-04-08 09:23:22 | segmentation_sanity_train

### Summary
- model: unet
- device: cpu
- epochs: 3
- train_samples: 2
- val_samples: 0
- batch_size: 2
- visuals_saved: 0
- best_val_dice: 0.0
- last_val_dice: 0.0

### Details
- output_root: outputs\seg_sanity\20260408_092320
- history_csv: outputs\seg_sanity\20260408_092320\history.csv
- checkpoint: outputs\seg_sanity\20260408_092320\checkpoint_last.pt
- visuals_dir: outputs\seg_sanity\20260408_092320\visuals


## 2026-04-08 09:24:16 | segmentation_sanity_train

### Summary
- model: unet
- device: cpu
- epochs: 3
- train_samples: 2
- val_samples: 0
- batch_size: 2
- visuals_saved: 0
- best_val_dice: 0.0
- last_val_dice: 0.0

### Details
- output_root: outputs\seg_sanity\20260408_092415
- history_csv: outputs\seg_sanity\20260408_092415\history.csv
- checkpoint: outputs\seg_sanity\20260408_092415\checkpoint_last.pt
- visuals_dir: outputs\seg_sanity\20260408_092415\visuals


## 2026-04-08 09:38:37 | metadata_refresh

### Summary
- metadata_rows: 2674
- failed_rows: 2
- train_rows: 1871
- val_rows: 401
- test_rows: 402

### Details
- metadata_csv: D:\Kuliah\S2ElektronikaITS\Tesis\Riset\GenerativeAI\data\processed\metadata.csv
- failed_csv: D:\Kuliah\S2ElektronikaITS\Tesis\Riset\GenerativeAI\data\processed\failed_samples.csv
- split_info: D:\Kuliah\S2ElektronikaITS\Tesis\Riset\GenerativeAI\data\outputs\split_info.json


## 2026-04-08 09:48:01 | segmentation_sanity_train

### Summary
- model: unet
- device: cpu
- epochs: 3
- train_samples: 192
- val_samples: 96
- batch_size: 2
- visuals_saved: 12
- best_val_dice: 0.04969728274279509
- last_val_dice: 0.04969728274279509

### Details
- output_root: outputs\seg_sanity\20260408_094502
- history_csv: outputs\seg_sanity\20260408_094502\history.csv
- checkpoint: outputs\seg_sanity\20260408_094502\checkpoint_last.pt
- visuals_dir: outputs\seg_sanity\20260408_094502\visuals


## 2026-04-08 11:32:19 | segmentation_sanity_train

### Summary
- model: unet
- device: cpu
- epochs: 3
- train_samples: 192
- val_samples: 96
- batch_size: 2
- visuals_saved: 12
- best_val_dice: 0.06486189893136422
- last_val_dice: 0.06486189893136422

### Details
- output_root: outputs\seg_sanity\20260408_112712
- history_csv: outputs\seg_sanity\20260408_112712\history.csv
- checkpoint: outputs\seg_sanity\20260408_112712\checkpoint_last.pt
- visuals_dir: outputs\seg_sanity\20260408_112712\visuals


## 2026-04-08 11:32:51 | segmentation_sanity_train

### Summary
- model: unet
- device: cpu
- epochs: 3
- train_samples: 16
- val_samples: 8
- batch_size: 4
- visuals_saved: 8
- best_val_dice: 0.021502542309463024
- last_val_dice: 0.019534213934093714

### Details
- output_root: outputs\seg_sanity\20260408_113224
- history_csv: outputs\seg_sanity\20260408_113224\history.csv
- checkpoint: outputs\seg_sanity\20260408_113224\checkpoint_last.pt
- visuals_dir: outputs\seg_sanity\20260408_113224\visuals
- debug_batch: [DEBUG val] logits_shape=(4, 1, 256, 256) mask_min=0.0000 mask_max=1.0000 mask_unique=[0.0, 1.0] dice_per_sample=[0.0176, 0.0278, 0.0405, 0.0187]


## 2026-04-08 16:42:14 | segmentation_sanity_train

### Summary
- model: unet
- device: cpu
- epochs: 3
- train_samples: 192
- val_samples: 96
- batch_size: 2
- visuals_saved: 12
- best_val_dice: 0.02570542828955998
- last_val_dice: 0.025583612402745832

### Details
- output_root: outputs\seg_sanity\20260408_164052
- history_csv: outputs\seg_sanity\20260408_164052\history.csv
- checkpoint: outputs\seg_sanity\20260408_164052\checkpoint_last.pt
- visuals_dir: outputs\seg_sanity\20260408_164052\visuals
- debug_batch: [DEBUG val] logits_shape=(2, 1, 256, 256) mask_min=0.0000 mask_max=1.0000 mask_unique=[0.0, 1.0] dice_per_sample=[0.0209, 0.0303]


## 2026-04-08 17:26:53 | segmentation_sanity_train

### Summary
- model: unet
- device: cpu
- epochs: 3
- train_samples: 192
- val_samples: 96
- batch_size: 2
- visuals_saved: 12
- best_val_dice: 0.18934935424476862
- last_val_dice: 0.18934935424476862

### Details
- output_root: outputs\seg_sanity\20260408_172541
- history_csv: outputs\seg_sanity\20260408_172541\history.csv
- checkpoint: outputs\seg_sanity\20260408_172541\checkpoint_last.pt
- visuals_dir: outputs\seg_sanity\20260408_172541\visuals
- debug_batch: [DEBUG val] logits_shape=(2, 1, 256, 256) mask_min=0.0000 mask_max=1.0000 mask_unique=[0.0, 1.0] dice_per_sample=[0.0572, 0.085]


## 2026-04-08 18:00:06 | segmentation_sanity_train

### Summary
- model: unet
- device: cpu
- loss: bce_softdice
- pos_weight: 50.0
- loss_alpha: 0.5
- epochs: 3
- train_samples: 192
- val_samples: 96
- batch_size: 2
- visuals_saved: 12
- best_val_dice: 0.02807540954866757
- last_val_dice: 0.02807540954866757

### Details
- output_root: outputs\seg_sanity\20260408_175910
- history_csv: outputs\seg_sanity\20260408_175910\history.csv
- checkpoint: outputs\seg_sanity\20260408_175910\checkpoint_last.pt
- visuals_dir: outputs\seg_sanity\20260408_175910\visuals
- debug_batch: [DEBUG val] logits_shape=(2, 1, 256, 256) mask_min=0.0000 mask_max=1.0000 mask_unique=[0.0, 1.0] dice_per_sample=[0.0224, 0.033]

