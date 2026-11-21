import os
import sys
import tempfile
import unittest
from unittest.mock import patch
from torch.utils.data import DataLoader

from evaluate import UCF101FullTestDataset
from dataset import UCF101Dataset


class TestEvaluateEdgeCases(unittest.TestCase):
    def test_missing_videos_aborts_with_error(self):
        with tempfile.TemporaryDirectory() as td:
            data_root = os.path.join(td, "data")
            split_file = os.path.join(td, "fake_split.txt")
            os.makedirs(data_root, exist_ok=True)
            
            with open(split_file, "w") as f:
                f.write("ClassA/v_ClassA_g01_c01 100 0\n")
                f.write("ClassB/v_ClassB_g01_c01 100 1\n")
                
            with self.assertRaises(FileNotFoundError) as ctx:
                _ = UCF101FullTestDataset(data_root=data_root, split_file=split_file)
            self.assertIn("2/2 evaluation videos are missing", str(ctx.exception))
            print("  [OK] Successfully caught FileNotFoundError for missing evaluation videos.")

    def test_corrupted_video_raises_runtime_error(self):
        with tempfile.TemporaryDirectory() as td:
            data_root = os.path.join(td, "data")
            video_dir = os.path.join(data_root, "videos", "UCF-101", "ClassA")
            os.makedirs(video_dir, exist_ok=True)
            fake_video = os.path.join(video_dir, "v_ClassA_g01_c01.avi")
            with open(fake_video, "w") as f:
                f.write("corrupted content")
            
            split_file = os.path.join(td, "fake_split.txt")
            with open(split_file, "w") as f:
                f.write("ClassA/v_ClassA_g01_c01 100 0\n")
                
            dataset = UCF101FullTestDataset(data_root=data_root, split_file=split_file, eval_mode='single')
            self.assertEqual(len(dataset), 1)
            
            # Mock load_video_frames to simulate decoding error
            with patch("evaluate.load_video_frames", side_effect=IOError("Corrupted packet")):
                with self.assertRaises(RuntimeError) as ctx:
                    _ = dataset[0]
                self.assertIn("Unreadable evaluation video", str(ctx.exception))
                self.assertIn("Corrupted packet", str(ctx.exception))
                print("  [OK] Successfully caught RuntimeError on unreadable video without infinite recursion.")

    def test_dataloader_propagation_multiprocessing(self):
        with tempfile.TemporaryDirectory() as td:
            data_root = os.path.join(td, "data")
            video_dir = os.path.join(data_root, "videos", "UCF-101", "ClassA")
            os.makedirs(video_dir, exist_ok=True)
            fake_video = os.path.join(video_dir, "v_ClassA_g01_c01.avi")
            with open(fake_video, "w") as f:
                f.write("corrupted header bytes")
            
            split_file = os.path.join(td, "fake_split.txt")
            with open(split_file, "w") as f:
                f.write("ClassA/v_ClassA_g01_c01 100 0\n")
                
            dataset = UCF101FullTestDataset(data_root=data_root, split_file=split_file, eval_mode='single')
            loader = DataLoader(dataset, batch_size=1, num_workers=2)
            
            with self.assertRaises(RuntimeError) as ctx:
                for _ in loader:
                    pass
            self.assertIn("Fatal Evaluation Error", str(ctx.exception))
            print("  [OK] Successfully verified DataLoader worker exception propagation (num_workers=2).")

    def test_dataset_val_fail_stop(self):
        with tempfile.TemporaryDirectory() as td:
            data_root = os.path.join(td, "data")
            video_dir = os.path.join(data_root, "videos", "UCF-101", "ClassA")
            os.makedirs(video_dir, exist_ok=True)
            fake_video = os.path.join(video_dir, "v_ClassA_g01_c01.avi")
            with open(fake_video, "w") as f:
                f.write("corrupted content")
            
            split_file = os.path.join(td, "fake_split.txt")
            with open(split_file, "w") as f:
                f.write("ClassA/v_ClassA_g01_c01 100 0\n")
                
            dataset = UCF101Dataset(data_root=data_root, split_file=split_file, is_train=False)
            with patch("dataset.load_video_frames", side_effect=IOError("Corrupted frames")):
                with self.assertRaises(RuntimeError) as ctx:
                    _ = dataset[0]
                self.assertIn("Unreadable validation video", str(ctx.exception))
                print("  [OK] Successfully caught RuntimeError on validation dataset without fallback.")


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestEvaluateEdgeCases)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if not result.wasSuccessful():
        sys.exit(1)
