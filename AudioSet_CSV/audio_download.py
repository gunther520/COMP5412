import os
import pandas as pd
import subprocess

# Configuration
CSV_FILE = "balanced_train_segments.csv"  # Change to your CSV file
OUTPUT_DIR = "audioset_wavs"
YDL_OPTS = "--format bestaudio --extract-audio --audio-format wav --audio-quality 192K"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_video_ids(csv_file):
    """ Load YouTube video IDs from the AudioSet CSV file. """
    df = pd.read_csv(csv_file, skiprows=3)  # Skip first 3 header lines
    video_ids = df.iloc[:, 0].tolist()  # First column contains video IDs
    return video_ids

def download_audio(video_id):
    """ Download and extract audio from YouTube using yt-dlp. """
    youtube_url = f"https://www.youtube.com/watch?v={video_id}"
    output_path = os.path.join(OUTPUT_DIR, f"{video_id}.wav")
    
    command = f'yt-dlp {YDL_OPTS} -o "{output_path}" "{youtube_url}"'
    subprocess.run(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def main():
    """ Main function to download all audio files. """
    video_ids = load_video_ids(CSV_FILE)
    print(f"Found {len(video_ids)} videos. Starting audio download...")

    for idx, video_id in enumerate(video_ids[:10]):  # Limit to 10 for testing
        print(f"[{idx+1}/{len(video_ids)}] Downloading audio for {video_id}...")
        download_audio(video_id)

    print("Audio download complete!")

if __name__ == "__main__":
    main()
