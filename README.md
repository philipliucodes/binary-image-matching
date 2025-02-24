# Binary Template Matching

Binary Template Matching is a Python-based approach that utilizes **binarization** to transform input and template images, followed by direct pixel matching for efficient and accurate template detection.

Binarization enhances template matching by simplifying images to their fundamental shapes, making it especially useful when the template's shading, contrast, or color may vary. This technique is effective for locating objects with distinct contours, such as identifying a cursor in a screen recording, detecting symbols in documents, or recognizing objects under varying lighting conditions. By reducing the complexity of image data, binarization improves robustness in situations where traditional grayscale or color-based matching might fail.

## Installation

Clone the repository and install the required dependencies using `pip`:

```bash
git clone https://github.com/yourusername/binary-template-matching.git
cd binary-template-matching
pip install -r requirements.txt
```

## Usage

### Extracting Frames from a Video

The `frame_extractor.py` script extracts a specific frame from a video at a given timestamp.

#### Command-line Usage

```bash
python src/frame_extractor.py path/to/video.mp4 MM:SS.MS --output frames/
```

#### Example

Extract a frame at **00:44.250** from `video.mp4` and save it to the `extracted_frames/` directory:

```bash
python src/frame_extractor.py sample_video.mp4 00:44.250 --output extracted_frames
```

---

### Template Matching with Binarization

The `template_matcher.py` script allows template matching using binarized images. It supports both **single image** and **directory-based** matching.

#### Command-line Usage

```bash
python src/template_matcher.py path/to/video.mp4 template_image_or_directory --interval 5.0 --confidence_threshold 0.90 --white_threshold 200 --output results/ --csv results/match_results.csv --save_bboxes --search_width 100 --search_height 100
```

#### Examples

1. **Match frames extracted from a video against templates at 10ms intervals, searching within a 100x100 pixel region around the last match:**

   ```bash
   python src/template_matcher.py video.mp4 templates/ --interval 0.01 --confidence_threshold 0.90 --white_threshold 200 --output results/ --csv results/match_results.csv --save_bboxes --search_width 100 --search_height 100
   ```

2. **Match frames extracted every 600 seconds against templates, with a larger 200x200 pixel search area around previous matches:**

   ```bash
   python src/template_matcher.py video.mp4 templates/ --interval 600 --confidence_threshold 0.90 --white_threshold 200 --output results/ --csv results/match_results.csv --save_bboxes --search_width 200 --search_height 200
   ```

3. **Match a single image against a single template without using a predefined search region:**

   ```bash
   python src/template_matcher.py input.jpg template.jpg --confidence_threshold 0.90 --white_threshold 200 --output results/
   ```

4. **Match a single image against multiple templates (inside a directory), allowing a 150x150 pixel search area for optimization:**

   ```bash
   python src/template_matcher.py input.jpg templates/ --confidence_threshold 0.90 --white_threshold 200 --output results/ --search_width 150 --search_height 150
   ```

5. **Match multiple input images against a single template:**

   ```bash
   python src/template_matcher.py images/ template.jpg --confidence_threshold 0.90 --white_threshold 200 --output results/
   ```

6. **Match multiple input images against multiple templates with a tighter 75x75 pixel search window:**

   ```bash
   python src/template_matcher.py images/ templates/ --confidence_threshold 0.90 --white_threshold 200 --output results/ --search_width 75 --search_height 75
   ```

---

## Testing Data

A sample dataset from **Lee Memory and Cognition Lab at Purdue University**, containing cursor template images along with the screen recordings they were extracted from, is available for testing.

[Download Dataset Here](https://drive.google.com/drive/folders/1z6H-jSOXbFHEh0YNDAoG9efRs2j1vyI7?usp=drive_link)
