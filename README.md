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
python src/template_matcher.py path/to/video.mp4 template_image_or_directory --interval 0.01 --confidence_threshold 0.90 --white_threshold 200 --output results/ --csv results/match_results.csv --save_bboxes
```

#### Examples

1. **Match frames extracted from a video against templates at 10ms intervals:**

   ```bash
   python src/template_matcher.py video.mp4 templates/ --interval 0.01 --confidence_threshold 0.90 --white_threshold 200 --output results/ --csv results/match_results.csv --save_bboxes
   ```

2. **Match frames extracted every 600 seconds against templates:**

   ```bash
   python src/template_matcher.py video.mp4 templates/ --interval 600 --confidence_threshold 0.90 --white_threshold 200 --output results/ --csv results/match_results.csv --save_bboxes
   ```

3. **Match a single image against a single template:**

   ```bash
   python src/template_matcher.py input.jpg template.jpg --confidence_threshold 0.90 --white_threshold 200 --output results/
   ```

4. **Match a single image against multiple templates (inside a directory):**

   ```bash
   python src/template_matcher.py input.jpg templates/ --confidence_threshold 0.90 --white_threshold 200 --output results/
   ```

5. **Match multiple input images against a single template:**

   ```bash
   python src/template_matcher.py images/ template.jpg --confidence_threshold 0.90 --white_threshold 200 --output results/
   ```

6. **Match multiple input images against multiple templates:**

   ```bash
   python src/template_matcher.py images/ templates/ --confidence_threshold 0.90 --white_threshold 200 --output results/
   ```

---

## Testing Data

A sample dataset from **Lee Memory and Cognition Lab at Purdue University**, containing cursor template images along with the screen recordings they were extracted from, is available for testing.

[Download Dataset Here](https://drive.google.com/drive/folders/1z6H-jSOXbFHEh0YNDAoG9efRs2j1vyI7?usp=drive_link)
