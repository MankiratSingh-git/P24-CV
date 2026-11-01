import os
import sys
import argparse
import time
import zipfile
import urllib.request
import subprocess
import shutil
import numpy as np
import cv2

HF_URL = "https://huggingface.co/datasets/quchenyuan/UCF101-ZIP/resolve/main/UCF-101.zip"
EXPECTED_BYTES = 6957373664  # 6.48 GB


def download_chunk(url, part_path, current_bytes, max_retries=50):
    """
    Downloads remaining bytes from current_bytes onwards with automatic reconnection.
    """
    retry_count = 0
    while retry_count < max_retries:
        if os.path.exists(part_path):
            current_bytes = os.path.getsize(part_path)
        else:
            current_bytes = 0

        if current_bytes >= EXPECTED_BYTES:
            print("\nDownload complete (100%).")
            return True

        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        if current_bytes > 0:
            headers['Range'] = f"bytes={current_bytes}-"
            print(f"\n[Resume] Reconnecting from byte {current_bytes} ({current_bytes / (1024**3):.2f}/{EXPECTED_BYTES / (1024**3):.2f} GB)...")
        else:
            print(f"\nStarting fresh download of {EXPECTED_BYTES / (1024**3):.2f} GB...")

        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                status = response.status
                mode = 'ab' if current_bytes > 0 and status == 206 else 'wb'
                if mode == 'wb':
                    current_bytes = 0

                downloaded = current_bytes
                start_time = time.time()
                last_print_time = start_time

                with open(part_path, mode) as f:
                    block_size = 1024 * 1024  # 1 MB blocks
                    while True:
                        chunk = response.read(block_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)

                        curr_time = time.time()
                        if curr_time - last_print_time >= 1.0:
                            elapsed = curr_time - start_time
                            speed = (downloaded - current_bytes) / (elapsed + 1e-6) / (1024 * 1024)
                            percent = (downloaded / EXPECTED_BYTES * 100.0)
                            print(
                                f"\rProgress: {downloaded / (1024**3):.2f}/{EXPECTED_BYTES / (1024**3):.2f} GB "
                                f"({percent:.1f}%) | Speed: {speed:.2f} MB/s",
                                end="", flush=True
                            )
                            last_print_time = curr_time

                # Check if entire file was received
                if os.path.getsize(part_path) >= EXPECTED_BYTES:
                    print("\nDownload completed successfully.")
                    return True
                else:
                    print(f"\n[Notice] Connection closed before reaching total size ({os.path.getsize(part_path)} / {EXPECTED_BYTES} bytes). Retrying...")

        except Exception as e:
            retry_count += 1
            print(f"\n[Warning] Network interruption ({e}). Retrying ({retry_count}/{max_retries}) in 5s...")
            time.sleep(5)

    return False


def download_with_curl(url, dest_path):
    """
    Alternative downloader using system curl.exe with native resume (-C -).
    """
    curl_path = shutil.which("curl.exe") or shutil.which("curl")
    if not curl_path:
        return False

    print(f"Using system curl ({curl_path}) with automatic resume...")
    part_path = dest_path + ".part"
    cmd = [
        curl_path,
        "-C", "-",
        "-L",
        "--retry", "50",
        "--retry-delay", "3",
        "--retry-connrefused",
        "-o", part_path,
        url
    ]
    try:
        ret = subprocess.run(cmd)
        if ret.returncode == 0 and os.path.exists(part_path) and os.path.getsize(part_path) >= EXPECTED_BYTES:
            if os.path.exists(dest_path):
                os.remove(dest_path)
            os.rename(part_path, dest_path)
            return True
    except Exception as e:
        print(f"curl invocation error: {e}")
    return False


def extract_zip(zip_path, extract_to):
    """Extract zip archive with progress feedback."""
    print(f"Validating archive: {zip_path}...")
    if not zipfile.is_zipfile(zip_path):
        raise ValueError(f"File {zip_path} is not a valid or complete zip archive!")

    print(f"Extracting to: {extract_to}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        infolist = zip_ref.infolist()
        total_files = len(infolist)
        for i, member in enumerate(infolist):
            zip_ref.extract(member, extract_to)
            if (i + 1) % 500 == 0 or (i + 1) == total_files:
                print(f"\rExtracted {i+1}/{total_files} files ({(i+1)/total_files*100:.1f}%)...", end="", flush=True)
    print("\nExtraction complete.")


def create_synthetic_dataset(target_dir, num_classes=101, videos_per_class=5, frames_per_video=64):
    """
    Creates a lightweight synthetic dataset with real .avi video clips so users
    can verify training pipelines immediately without downloading the full 6.5 GB.
    """
    print(f"Creating lightweight synthetic dataset in {target_dir} ({num_classes} classes)...")
    videos_dir = os.path.join(target_dir, 'videos')
    os.makedirs(videos_dir, exist_ok=True)

    class_indices_file = os.path.join("data", "splits", "class_indices.txt")
    class_names = []
    if os.path.exists(class_indices_file):
        with open(class_indices_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    class_names.append(parts[1])
    else:
        class_names = [f"ActionClass_{i:03d}" for i in range(num_classes)]

    class_names = class_names[:num_classes]
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')

    for c_idx, c_name in enumerate(class_names):
        c_dir = os.path.join(videos_dir, c_name)
        os.makedirs(c_dir, exist_ok=True)

        for v_idx in range(videos_per_class):
            v_name = f"v_{c_name}_g{v_idx+1:02d}_c01.avi"
            v_path = os.path.join(c_dir, v_name)
            if os.path.exists(v_path):
                continue

            out = cv2.VideoWriter(v_path, fourcc, 25.0, (320, 240))
            base_color = ((c_idx * 37) % 255, (c_idx * 73) % 255, (c_idx * 131) % 255)
            for f_idx in range(frames_per_video):
                frame = np.full((240, 320, 3), base_color, dtype=np.uint8)
                cx = int(50 + 200 * (f_idx / float(frames_per_video)))
                cy = int(120 + 30 * np.sin(f_idx * 0.2))
                cv2.circle(frame, (cx, cy), 20, (255, 255, 255), -1)
                out.write(frame)
            out.release()

        print(f"\rCreated classes: {c_idx+1}/{len(class_names)}", end="", flush=True)

    print("\nSynthetic dataset successfully created!")


def main():
    parser = argparse.ArgumentParser(description="Download or setup UCF-101 dataset")
    parser.add_argument("--dest", type=str, default="data/ucf101", help="Destination folder")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic video dataset for instant pipeline testing")
    parser.add_argument("--use-curl", action="store_true", help="Use system curl.exe for downloading")
    parser.add_argument("--extract-only", type=str, default=None, help="Extract existing zip file")
    args = parser.parse_args()

    dest_dir = os.path.abspath(args.dest)
    os.makedirs(dest_dir, exist_ok=True)

    if args.synthetic:
        create_synthetic_dataset(dest_dir, num_classes=101, videos_per_class=5, frames_per_video=64)
        return

    if args.extract_only:
        extract_zip(args.extract_only, os.path.join(dest_dir, "videos"))
        return

    zip_path = os.path.join(dest_dir, "UCF-101.zip")
    part_path = zip_path + ".part"

    # Check if a prematurely completed or corrupted zip exists
    if os.path.exists(zip_path):
        if os.path.getsize(zip_path) < EXPECTED_BYTES or not zipfile.is_zipfile(zip_path):
            print(f"[Notice] Existing {zip_path} is incomplete ({os.path.getsize(zip_path)}/{EXPECTED_BYTES} bytes).")
            print("Renaming to .part to resume downloading from where it stopped...")
            if os.path.exists(part_path):
                os.remove(part_path)
            os.rename(zip_path, part_path)

    if not os.path.exists(zip_path):
        print(f"Target file: {zip_path}")
        success = False

        if args.use_curl:
            success = download_with_curl(HF_URL, zip_path)

        if not success:
            initial_bytes = os.path.getsize(part_path) if os.path.exists(part_path) else 0
            success = download_chunk(HF_URL, part_path, initial_bytes, max_retries=50)
            if success:
                if os.path.exists(zip_path):
                    os.remove(zip_path)
                os.rename(part_path, zip_path)

        if not success:
            print("\nDownload could not be completed. You can re-run the command to resume from where it stopped.")
            sys.exit(1)

    extract_to = os.path.join(dest_dir, "videos")
    extract_zip(zip_path, extract_to)


if __name__ == "__main__":
    main()
