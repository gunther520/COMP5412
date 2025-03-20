import os
import pandas as pd
import subprocess

# Configuration
CSV_FILE = "AudioData/AudioSet_CSV/balanced_train_segments.csv"  # Change to your CSV file
OUTPUT_DIR = "AudioData/AudioSet_CSV/audioset_classes"
TARGET_CLASSES = {
    "/m/03wvsk":"Hair_dryer",
    "/m/0316dw":"Typing",
    "/m/02x984l":"Fan",
    "/m/06mb1":"Rain",
    "/m/0j2kx":"Waterfall",
    "/t/dd00092":"Wind_noise",
    "/m/0chx_":"White_noise",
    "/m/01jwx6":"Vibration",
    "/m/03w41f":"Church_bell",
    "/m/0gy1t2s":"Bicycle_bell",
    "/m/01b82r":"Sawing",
    "/m/07qn4z3":"Rattle",
    "/m/07s34ls":"Whir",
    "/m/09ddx":"Duck",
    "/m/07qdb04":"Quack",
    "/m/0dbvp":"Goose",
    "/m/07qwf61":"Honk",
    "/m/0j2kx":"Waterfall",
    "/m/0g6b5":"Fireworks",
    "/m/07qb_dv":"Scratch",

}

YDL_OPTS = "--format bestaudio --extract-audio --audio-format wav --audio-quality 0"
MAX_DOWNLOADS = 200  # User-defined number of audio segments to download

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_video_ids(csv_file):
    """ Load YouTube video IDs and annotations from the AudioSet CSV file, skipping the first comment lines. """
    try:
        # Skip the 3 comment lines
        with open(csv_file, 'r') as f:
            lines = f.readlines()[3:]
        
        # Create empty lists for each column
        ytids = []
        start_times = []
        end_times = []
        labels = []
        
        # Parse each line manually
        for line in lines:
            if line.strip():  # Skip empty lines
                parts = line.strip().split(', ', 3)  # Split by first three commas only
                if len(parts) == 4:
                    ytids.append(parts[0])
                    start_times.append(float(parts[1]))
                    end_times.append(float(parts[2]))
                    labels.append(parts[3])
        
        # Create dataframe
        df = pd.DataFrame({
            'YTID': ytids,
            'start_seconds': start_times,
            'end_seconds': end_times,
            'positive_labels': labels
        })
        
    except Exception as e:
        print(f"Error parsing the CSV file: {e}")
        return None
    
    return df

def download_audio(video_id, start_time, end_time, class_name):
    """ Download and extract audio for a specific segment from YouTube using yt-dlp. """
    youtube_url = f"https://www.youtube.com/watch?v={video_id}"
    class_dir = os.path.join(OUTPUT_DIR, class_name)
    os.makedirs(class_dir, exist_ok=True)
    
    # Define output file path with WAV extension
    output_path = os.path.join(class_dir, f"{video_id}_{start_time}-{end_time}.wav")
    
    # Download the segment of audio for the specified time range
    command = f'yt-dlp {YDL_OPTS} -o "{output_path}" --download-sections "*{start_time}-{end_time}" "{youtube_url}"'
    
    # Show output to diagnose issues
    print(f"Executing command: {command}")
    try:
        subprocess.run(command, shell=True, check=True)
        print(f"Successfully downloaded {video_id}")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading {video_id}: {e}")

def main():
    """ Main function to process all video IDs and download specific audio segments. """
    df = load_video_ids(CSV_FILE)
    
    if df is None:
        print("Exiting due to CSV loading issues.")
        return

    print(f"Found {len(df)} rows. Starting audio download...")

    # Filter the dataset for target classes
    filtered_df = df[
        df['positive_labels'].str.contains('|'.join(TARGET_CLASSES.keys()))
    ]
    
    print(f"After filtering, found {len(filtered_df)} matching videos.")

    downloads_count = 0
    print("size of filtered_df: ", len(filtered_df))  
    for idx, row in filtered_df.iterrows():
        if downloads_count >= MAX_DOWNLOADS:
            print(f"Reached the download limit of {MAX_DOWNLOADS} audio segments. Stopping iteration.")
            break  # Stop processing further rows once the limit is reached

        video_id = row['YTID']
        start_time = row['start_seconds']
        end_time = row['end_seconds']
        class_labels = row['positive_labels'].split(',')
        
        downloaded=0
        # Download audio only for target classes
        for class_id, class_name in TARGET_CLASSES.items():
            if class_id in class_labels:
                print(f"[{downloads_count+1}/{MAX_DOWNLOADS}] Downloading audio for {video_id} ({class_name})...")
                download_audio(video_id, start_time, end_time, class_name)
                downloaded=1

        downloads_count += downloaded
        print(f"downloads_count: {downloads_count}")

    print("Audio download complete!")

if __name__ == "__main__":
    main()
