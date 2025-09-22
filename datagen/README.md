## Aerial‑D: Dataset Generation Pipeline

Build the Aerial‑D referring expression dataset for aerial imagery using iSAID and LoveDA. The pipeline crops images into patches, adds spatial/relational rules, generates expressions, filters for uniqueness, and optionally applies historic effects and LLM enhancement.

### Setup
- Install dependencies from the repo root: `pip install -r ../requirements.txt`
- Place raw datasets:
  - iSAID under `datagen/isaid/`
  - LoveDA under `datagen/LoveDA/`
  - (Optional) DeepGlobe roads under `datagen/`

### Run the full pipeline
```bash
cd datagen
./pipeline/run_pipeline.sh
# Sampling and utilities
./pipeline/run_pipeline.sh --num_images 100           # sample run
./pipeline/run_pipeline.sh --clean --zip              # clean previous output + create zip
```

Supported flags:
- `--num_images N` select N images per split (iSAID, LoveDA, DeepGlobe)
- `--start_image_id X` / `--end_image_id Y` to bound IDs
- `--num_workers W` worker parallelism for cropping
- `--random_seed S` reproducible sampling
- `--clean` delete `dataset/` before starting
- `--zip` create an archive of the final dataset

### What the pipeline does
1) iSAID patches → 480×480 patches with overlap and complete instances  
2) LoveDA patches → compatible patch extraction  
3) (Optional) DeepGlobe roads → road instances and XML annotations  
4) Add rules → 3×3 grid position, pairwise relations, extreme positions, size attributes, grouping  
5) Generate expressions → category, position, relation, extreme, size, combinations  
6) Filter unique → keep only expressions that refer to exactly one target  
7) Historic filter → optional B&W/sepia/noise variants

### Outputs
```
datagen/dataset/
├── train/
│   ├── annotations/   # XML with RLE masks + expressions
│   └── images/        # 480×480 PNG patches
└── val/
    ├── annotations/
    └── images/
```
Each XML contains bbox, segmentation (RLE), grid positions, relations, expression lists (original/enhanced/unique), and optional historic metadata.

### Viewers
```bash
python utils/app.py --split train --port 5001       # LLM‑enhanced viewer
python utils/rule_viewer.py --split val --port 5002 # rule‑based viewer
```

### Reproduce the Aerial‑D release
- Run the full pipeline to include uniqueness filtering and historic processing
- Optionally run LLM enhancement (see `../llm/README.md`) to add enhanced/unique expressions

### Notes
- Outputs are large; use `--num_images` during development
- If paths change, update dataset roots in scripts accordingly

