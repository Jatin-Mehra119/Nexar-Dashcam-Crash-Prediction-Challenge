# Nexar-Dashcam-Crash-Prediction-Challenge

# 🚗 VideoMAE-2 for Dashcam Collision Prediction

This repository contains all the data preprocessing, training and inference scripts used for predicting vehicle collisions using dashcam footage, developed for the [Nexar Dashcam Collision Prediction Challenge](https://www.kaggle.com/competitions/nexar-collision-prediction). The model achieved a
**11th place** finish on the private leaderboard with a score of **0.80** mean Average Precision (mAP).
[![ChatGPT Image May 10, 2025, 11_39_58 AM](https://github.com/user-attachments/assets/bda3bb70-dcb8-4b4d-995a-b67b7aecad73)](https://huggingface.co/jatinmehra/Accident-Detection-using-Dashcam)

----------

## 🧠 Model Overview

-   **Architecture**: [VideoMAE-2 Large](https://huggingface.co/MCG-NJU/videomae-large-finetuned-kinetics) fine-tuned for binary classification (collision/near-miss vs. normal driving).
    
-   **Feature Extraction**: Utilized [TimeSformer](https://huggingface.co/facebook/timesformer-base-finetuned-k400) for preprocessing input frames.
    
-   **Input**: 16 frames per video, each resized to 224x224 pixels.
    
-   **Output**: Probability score indicating the likelihood of a collision or near-miss event.
    

----------

## 📁 Dataset

The model was trained on the [Nexar Collision Prediction Dataset](https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction).

-   750 non-collision videos
    
-   400 collision videos
    
-   350 near-miss videos [arXiv](https://arxiv.org/html/2503.03848v1?utm_source=chatgpt.com)
    

Each video is annotated with:

-   **Event Type**: Collision, near-miss, or normal driving
    
-   **Event Time**: Timestamp of the (near-)collision
    
-   **Alert Time**: Earliest time the event could be predicted.
    

For more details, refer to the [dataset paper](https://arxiv.org/abs/2503.03848).

----------

## 🛠️ Preprocessing Pipeline

1.  **Frame Extraction**: Sampled 16 frames per video, focusing on the interval around the alert time.
    
2.  **Feature Extraction**: Applied TimeSformer feature extractor to obtain pixel values.
    
3.  **Data Augmentation**: Implemented transformations such as horizontal flip, rotation, color jitter, and resized cropping.
    
4.  **Normalization**: Used ImageNet mean and standard deviation for normalization.
    

----------

## 🏋️ Training Details

-   **Framework**: PyTorch with Hugging Face Transformers and Trainer API.
    
-   **Training Configuration**:
    
    -   Batch Size: 4
        
    -   Epochs: 15
        
    -   Learning Rate: 3e-5
        
    -   Weight Decay: 0.01
        
    -   Evaluation Strategy: Per epoch
        
    -   Metric for Best Model: Average Precision
        
-   **Hardware**: Trained on 2x NVIDIA T4 GPUs (~4.5 hours)
    

----------

## 📊 Evaluation Metrics

The model's performance was evaluated using Mean Average Precision (mAP) across different time-to-accident intervals:

-   500ms
    
-   1000ms
    
-   1500ms

The final score is the mean of the Average Precision (AP) values at these intervals, emphasizing early and accurate collision predictions



## 📚 Citation

If you use this model or dataset, please cite:
```
@misc{nexar2025dashcamcollisionprediction,
  title={Nexar Dashcam Collision Prediction Dataset and Challenge},
  author={Daniel C. Moura and Shizhan Zhu and Orly Zvitia},
  year={2025},
  eprint={2503.03848},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2503.03848}
}
```
