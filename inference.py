import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
from transformers import AutoModelForVideoClassification

# Dataset class
class PreprocessedDashcamDataset(Dataset):
    def __init__(self, df, preprocessed_dir, mode='test'):
        self.df = df
        self.preprocessed_dir = preprocessed_dir
        self.mode = mode
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        video_id = self.df['id'].iloc[idx]
        preprocessed_path = os.path.join(self.preprocessed_dir, f"{video_id}.npy")
        pixel_values = torch.from_numpy(np.load(preprocessed_path))
        return {'pixel_values': pixel_values, 'video_id': video_id}

# Load test data
test_df = pd.read_csv('/kaggle/input/nexar-collision-prediction/test.csv')
test_dataset = PreprocessedDashcamDataset(test_df, '/kaggle/input/new-approach-test-16fps/preprocessed_test_8frames/', mode='test')
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)

# Load fine-tuned model
model_name = "/kaggle/input/new-torch-model/trained_videomae_model/"
model = AutoModelForVideoClassification.from_pretrained(model_name)
model.cuda()
model.eval()

# Generate predictions
test_preds = []
test_ids = []
with torch.no_grad():
    for batch in test_loader:
        pixel_values = batch['pixel_values'].cuda()
        video_ids = batch['video_id']
        outputs = model(pixel_values=pixel_values)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        test_preds.extend(probs)
        test_ids.extend([str(int(vid)).zfill(5) for vid in video_ids])

# Create submission file
submission = pd.DataFrame({'id': test_ids, 'score': test_preds})
submission.to_csv('/kaggle/working/submission.csv', index=False)

print("Predictions generated. Submission file saved to /kaggle/working/submission.csv")