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
    """Generates a list of timestamps at the given interval, ensuring precision."""
    duration = get_video_duration(video_path)
    timestamps = []
    current_time = 0.0
    while current_time < duration:
        current_time = round(current_time, 3)  # Ensure precise rounding to milliseconds
        minutes = int(current_time // 60)
        seconds = int(current_time % 60)
        milliseconds = int(round((current_time - int(current_time)) * 1000))
        timestamps.append(f"{minutes:02}:{seconds:02}.{milliseconds:03}")
        current_time += interval
    return timestamps

def extract_frame(video_path, timestamp, output_dir):
    """Extracts a single frame from the video at the given timestamp using frame_extractor.py."""
    os.makedirs(output_dir, exist_ok=True)
    frame_path = os.path.join(output_dir, f"{os.path.basename(video_path).split('.')[0]}_{timestamp.replace(':', '_').replace('.', '_')}.bmp")
    subprocess.run(["python", "src/frame_extractor.py", video_path, timestamp, "--output", output_dir], check=True)
    
    if os.path.exists(frame_path):
        print(f"Frame extracted: {frame_path}")
    else:
        print(f"Warning: Frame {timestamp} was not found at expected path {frame_path}")
    
    return frame_path if os.path.exists(frame_path) else None

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

def process_frame(frame_path, template_images, confidence_threshold, white_threshold, csv_output, save_bboxes, output_dir, 
                  last_matched_template, last_match_position, search_width, search_height):
    """Processes a single frame for template matching, first checking if the template has not moved, then expanding the search region."""
    if not os.path.exists(frame_path):
        print(f"Error: Frame {frame_path} not found, skipping template matching.")
        return last_matched_template, last_match_position

    input_image = Image.open(frame_path).convert("RGBA")
    input_array = np.array(input_image)
    input_alpha = input_array[:, :, 3]
    input_transformed = transform_pixels(input_array, input_alpha, white_threshold)

    if last_matched_template and last_matched_template in template_images:
        template_images.remove(last_matched_template)
        template_images.insert(0, last_matched_template)

    best_template = None
    best_match_position = (None, None)
    best_match_percentage = 0.0

    for template_filename in template_images:
        template_image = Image.open(template_filename).convert("RGBA")
        template_array = np.array(template_image)
        template_alpha = template_array[:, :, 3]
        template_transformed = transform_pixels(template_array, template_alpha, white_threshold)

        ih, iw = input_transformed.shape
        th, tw = template_transformed.shape

        search_regions = []

        # **STEP 1: Check the exact last match position first**
        if last_match_position is not None:
            x_prev, y_prev = last_match_position

            # Ensure the previous match position is within bounds
            if 0 <= x_prev <= iw - tw and 0 <= y_prev <= ih - th:
                search_regions.append((x_prev, x_prev, y_prev, y_prev))  # Check only this pixel location

        # **STEP 2: If not found, check the surrounding region**
        if last_match_position is not None:
            x_prev, y_prev = last_match_position
            x_start = max(0, x_prev - search_width // 2)
            x_end = min(iw - tw, x_prev + search_width // 2)
            y_start = max(0, y_prev - search_height // 2)
            y_end = min(ih - th, y_prev + search_height // 2)
            search_regions.append((x_start, x_end, y_start, y_end))

        # **STEP 3: If still not found, perform a full-frame search**
        search_regions.append((0, iw - tw, 0, ih - th))

        for x_start, x_end, y_start, y_end in search_regions:
            for y in range(y_start, y_end + 1):
                for x in range(x_start, x_end + 1):
                    roi = input_transformed[y:y+th, x:x+tw]
                    mask = template_alpha > 0
                    total_pixels = np.count_nonzero(mask)

                    if total_pixels > 0:
                        matching_pixels = np.sum(roi[mask] == template_transformed[mask])
                        match_score = matching_pixels / total_pixels

                        if match_score > confidence_threshold:
                            best_template = os.path.basename(template_filename)
                            best_match_position = (x, y)
                            best_match_percentage = match_score * 100
                            break
                if best_template:
                    break
            if best_template:
                break

    with open(csv_output, mode='a', newline='') as file:
        writer = csv.writer(file)
        if best_template:
            writer.writerow([os.path.basename(frame_path), best_template, best_match_position[0], best_match_position[1], f"{best_match_percentage:.2f}"])
            print(f"Frame '{os.path.basename(frame_path)}': Best template '{best_template}' at ({best_match_position[0]}, {best_match_position[1]}) with {best_match_percentage:.2f}% match.")
            last_matched_template = template_filename
            last_match_position = best_match_position

            if save_bboxes:
                result_array = np.array(input_image)
                cv2.rectangle(result_array, best_match_position, (best_match_position[0] + tw, best_match_position[1] + th), (255, 0, 0, 255), 2)
                output_image = Image.fromarray(result_array)
                output_path = os.path.join(output_dir, os.path.basename(frame_path))
                output_image.save(output_path)
        else:
            writer.writerow([os.path.basename(frame_path), "No match", "N/A", "N/A", "0.00"])
            print(f"Frame '{os.path.basename(frame_path)}': No template matches found.")
            last_matched_template = None
            last_match_position = None

    os.remove(frame_path)
    print(f"Deleted frame: {frame_path}")
    return last_matched_template, last_match_position

def template_matcher(video_path, template_path, interval, confidence_threshold, white_threshold, output_dir, csv_output, save_bboxes, search_width, search_height):
    """Extracts frames one at a time, performs template matching efficiently, and deletes frames after processing."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(csv_output), exist_ok=True)

    timestamps = generate_timestamps(video_path, interval)
    template_images = [os.path.join(template_path, f) for f in sorted(os.listdir(template_path)) if is_image(f)]

    if not template_images:
        print(f"Error: No valid template images found in '{template_path}'")
        return

    with open(csv_output, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["frame_name", "best_template", "match_x", "match_y", "match_percentage"])

    last_matched_template = None
    last_match_position = None  

    for timestamp in timestamps:
        frame_path = extract_frame(video_path, timestamp, output_dir)
        if frame_path:
            last_matched_template, last_match_position = process_frame(
                frame_path, template_images, confidence_threshold, white_threshold, csv_output, save_bboxes, output_dir, last_matched_template, last_match_position, search_width, search_height
            )

def main():
    """Parses command-line arguments and runs the template matcher on extracted video frames."""
    parser = argparse.ArgumentParser(description="Template matching on extracted video frames.")
    parser.add_argument("video_path", type=str, help="Path to input video file.")
    parser.add_argument("template_path", type=str, help="Path to template image or directory.")
    parser.add_argument("--interval", type=float, default=5, help="Interval in seconds between extracted frames.")
    parser.add_argument("--output", type=str, default="output", help="Directory to save matched images and CSV (default: output/)")
    parser.add_argument("--csv", type=str, default="output/match_results.csv", help="CSV file to store match results (default: output/match_results.csv)")
    parser.add_argument("--save_bboxes", action='store_true', help="Flag to save images with bounding boxes (default: False)")
    parser.add_argument("--search_width", type=int, default=100, help="Width of the region to search around the last matched position (default: 100)")
    parser.add_argument("--search_height", type=int, default=100, help="Height of the region to search around the last matched position (default: 100)")
    args = parser.parse_args()
    
    template_matcher(args.video_path, args.template_path, args.interval, 0.90, 200, args.output, args.csv, args.save_bboxes, args.search_width, args.search_height)

if __name__ == "__main__":
    main()
