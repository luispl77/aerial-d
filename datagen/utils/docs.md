# Dataset Generation Pipeline

This document provides a detailed overview of the two pipelines contained in this repository:

* **Rule‑based pipeline** (`pipeline/`)
* **LLM enhancement pipeline** (`llm/`)

Both operate on the iSAID dataset to produce a set of XML annotations and patch images.

## 1. Rule-based Pipeline

### Step 1: Create Patches

Script: `pipeline/1_create_patches.py`

* Reads the iSAID images and slides a `480x480` window with 20% overlap.
* Skips patches that contain more than 50% black pixels.
* Objects that intersect a patch by less than 50% are marked with an `is_cutoff` flag and their bounding boxes are recomputed from the cropped mask.
* Supports optional selection of images (`--num_images`, `--start_image_id`, `--end_image_id`) and multiprocessing across all CPU cores.
* Results are written to `dataset/patches/<split>/{images,annotations}`.

### Step 2: Add Rules

Script: `pipeline/2_add_rules.py`

* Parses each patch annotation and assigns spatial labels using a 3×3 grid. The center grid has a "no‑man's land" controlled by `alpha=0.2`.
* Detects borderline objects and records all possible positions.
* Determines extreme positions (topmost, bottommost, leftmost, rightmost) when separated by 5% of the image size.
* Calculates size relationships when one instance is `1.5x` larger or smaller than another.
* Groups nearby instances using DBSCAN (`eps=40`, `min_samples=2`, `max_samples=8`) and stores group relationships.
* Analyzes instance colors from the image and marks ambiguous colors when there are not enough pixels or low saturation.
* The updated XML files are saved to `dataset/patches_rules/<split>/annotations`.

### Step 3: Generate All Expressions

Script: `pipeline/3_generate_all_expressions.py`

* Builds many referring expressions for every instance and group.
* Expressions combine category names with grid position, relationships, extreme positions, size attributes and color terms.
* All expressions are written back to each XML file under an `<expressions>` node.
* Output directory: `dataset/patches_rules_expressions/<split>/annotations`.

### Step 4: Filter Unique Expressions

Script: `pipeline/4_filter_unique.py`

* Standardizes class names (e.g. "Large_Vehicle" → "large vehicle") and removes any expression that appears for more than one object.
* Color phrases are dropped for objects whose color was marked as ambiguous.
* Instances or groups with no remaining expressions are removed together with the corresponding patch image.
* Cleaned annotations are stored in `dataset/patches_rules_expressions_unique/<split>/annotations`.

## 2. LLM Enhancement Pipeline

### Step 1: Prepare Batch JSONL

Script: `llm/1_prepare_batch_jsonl.py`

* Reads the filtered XML files and for each object or group collects its original expressions.
* A 384×384 crop of the patch is uploaded to `gs://aerial-bucket/batch_inference/images` and referenced in a JSONL request.
* Each request asks Gemini to produce `NUM_ENHANCED=2` language variations for every original expression and `NUM_UNIQUE=2` new unique expressions.
* JSONL files are written to `dataset/batch_prediction/batch_prediction_requests_{train,val}.jsonl`.

### Step 2: Run Batch Prediction

Script: `llm/2_run_batch_prediction.py`

* Submits the JSONL files to Vertex AI using the `gemini-2.0-flash-001` model.
* Polls the job until it reaches a terminal state and downloads the result JSONL files to `dataset/batch_prediction`.

### Step 3: Parse Batch Results

Script: `llm/3_parse_batch_results.py`

* Reads the prediction JSONL files and inserts the generated data into each patch annotation.
* Adds `<enhanced_expressions>`, `<unique_description>` and `<unique_expressions>` elements.
* Final XML files are saved under `dataset/patches_rules_expressions_unique_llm/<split>/annotations`.

---

The resulting dataset contains patch images and rich annotations ready for model training.
