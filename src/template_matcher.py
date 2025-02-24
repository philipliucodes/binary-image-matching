import os
import argparse
import numpy as np
import csv
from PIL import Image
import cv2
import subprocess
import ffmpeg

def is_image(file_path):
    """Checks if the file is an image based on its extension."""
    return file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))

def get_video_duration(video_path):
    """Retrieves the duration of the video in seconds."""
    probe = ffmpeg.probe(video_path)
    duration = float(probe['format']['duration'])
    return duration

def generate_timestamps(video_path, interval):
    """Generates a list of timestamps at the given interval."""
    duration = get_video_duration(video_path)
    timestamps = []
    current_time = 0.0
    while current_time < duration:
        minutes = int(current_time // 60)
        seconds = int(current_time % 60)
        milliseconds = int((current_time - int(current_time)) * 1000)
        timestamps.append(f"{minutes:02}:{seconds:02}.{milliseconds:03}")
        current_time += interval
    return timestamps

def extract_frame(video_path, timestamp, output_dir):
    """Extracts a single frame from the video at the given timestamp using frame_extractor.py."""
    os.makedirs(output_dir, exist_ok=True)
    subprocess.run(["python", "src/frame_extractor.py", video_path, timestamp, "--output", output_dir], check=True)

def transform_pixels(image_array, alpha_channel, white_threshold):
    """Transforms an image to a binary mask based on transparency and whiteness."""
    transformed = np.zeros(image_array.shape[:2], dtype=np.uint8)
    non_transparent = alpha_channel > 0
    almost_white = (
        (image_array[:, :, 0] > white_threshold) &
        (image_array[:, :, 1] > white_threshold) &
        (image_array[:, :, 2] > white_threshold)
    )
    transformed[almost_white & non_transparent] = 255
    transformed[~almost_white & non_transparent] = 0
    return transformed

def template_matcher(video_path, template_path, interval, confidence_threshold, white_threshold, output_dir, csv_output):
    """Performs template matching on extracted video frames using the provided template(s)."""
    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    
    # Generate timestamps based on interval
    timestamps = generate_timestamps(video_path, interval)
    
    # Extract frames for each timestamp
    extracted_frames = []
    for timestamp in timestamps:
        extract_frame(video_path, timestamp, frames_dir)
    
    extracted_frames = [os.path.join(frames_dir, f) for f in sorted(os.listdir(frames_dir)) if is_image(f)]
    template_images = [os.path.join(template_path, f) for f in sorted(os.listdir(template_path)) if is_image(f)]
    
    if not extracted_frames:
        print(f"Error: No frames extracted from '{video_path}'")
        return
    if not template_images:
        print(f"Error: No valid template images found in '{template_path}'")
        return
    
    with open(csv_output, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["frame_name", "best_template", "match_x", "match_y", "match_percentage"])
    
        for input_filename in extracted_frames:
            input_image = Image.open(input_filename).convert("RGBA")
            input_array = np.array(input_image)
            input_alpha = input_array[:, :, 3]
            input_transformed = transform_pixels(input_array, input_alpha, white_threshold)
            
            best_template = None
            max_match_score = 0
            best_match_position = (None, None)
            best_match_percentage = 0.0
            
            for template_filename in template_images:
                template_image = Image.open(template_filename).convert("RGBA")
                template_array = np.array(template_image)
                template_alpha = template_array[:, :, 3]
                template_transformed = transform_pixels(template_array, template_alpha, white_threshold)
                
                ih, iw = input_transformed.shape
                th, tw = template_transformed.shape
                
                for y in range(ih - th + 1):
                    for x in range(iw - tw + 1):
                        roi = input_transformed[y:y+th, x:x+tw]
                        mask = template_alpha > 0
                        total_pixels = np.count_nonzero(mask)
                        
                        if total_pixels > 0:
                            matching_pixels = np.sum(roi[mask] == template_transformed[mask])
                            match_score = matching_pixels / total_pixels
                            
                            if match_score > max_match_score:
                                max_match_score = match_score
                                best_template = os.path.basename(template_filename)
                                best_match_position = (x, y)
                                best_match_percentage = match_score * 100
                
            if best_template:
                result_array = np.array(input_image)
                cv2.rectangle(result_array, best_match_position, (best_match_position[0] + tw, best_match_position[1] + th), (255, 0, 0, 255), 2)
                output_image = Image.fromarray(result_array)
                output_path = os.path.join(output_dir, os.path.basename(input_filename))
                output_image.save(output_path)
                
                writer.writerow([os.path.basename(input_filename), best_template, best_match_position[0], best_match_position[1], f"{best_match_percentage:.2f}"])
                print(f"Frame '{os.path.basename(input_filename)}': Best template '{best_template}' at ({best_match_position[0]}, {best_match_position[1]}) with {best_match_percentage:.2f}% match.")
            else:
                writer.writerow([os.path.basename(input_filename), "No match", "N/A", "N/A", "0.00"])
                print(f"Frame '{os.path.basename(input_filename)}': No template matches found.")

def main():
    """Parses command-line arguments and runs the template matcher on extracted video frames."""
    parser = argparse.ArgumentParser(description="Template matching on extracted video frames.")
    parser.add_argument("video_path", type=str, help="Path to input video file.")
    parser.add_argument("template_path", type=str, help="Path to template image or directory.")
    parser.add_argument("--interval", type=float, default=5, help="Interval in seconds between extracted frames.")
    parser.add_argument("--confidence_threshold", type=float, default=0.90, help="Threshold for matching confidence (default: 0.90)")
    parser.add_argument("--white_threshold", type=int, default=200, help="Threshold for defining near-white pixels (default: 200)")
    parser.add_argument("--output", type=str, default="output", help="Directory to save matched images and CSV (default: output/)")
    parser.add_argument("--csv", type=str, default="output/match_results.csv", help="CSV file to store match results (default: output/match_results.csv)")
    
    args = parser.parse_args()
    
    template_matcher(args.video_path, args.template_path, args.interval, args.confidence_threshold, args.white_threshold, args.output, args.csv)

if __name__ == "__main__":
    main()
