# Domain Style Transfer Testing

This script tests various methods to transform aerial images from one domain (e.g., iSAID) to look like another domain (e.g., DeepGlobe).

## Methods Implemented

### 1. **Simple Statistical Methods** (single reference image)
- **Histogram Matching**: Matches the histogram of source to target
- **Color Moment Transfer**: Matches mean and std of RGB channels
- **LAB Color Transfer**: Same as color moment but in LAB color space (better perceptual)

### 2. **Style Bank Methods** (multiple reference images)
- **Style Bank (RGB)**: Uses statistics computed from multiple target domain images in RGB space
- **Style Bank (LAB)**: Uses statistics computed from multiple target domain images in LAB space

### 3. **Neural Style Transfer** (implemented but needs training)
- **AdaIN**: Adaptive Instance Normalization approach (basic implementation included)

## Quick Usage

### Test iSAID → DeepGlobe transformation:
```bash
cd /cfs/home/u035679/aerialseg/clipsam/utils
python test_style_transfer.py --source_domain iSAID --target_domain DeepGlobe
```

### Test DeepGlobe → iSAID transformation:
```bash
python test_style_transfer.py --source_domain DeepGlobe --target_domain iSAID
```

### Build style banks first (recommended):
```bash
python test_style_transfer.py --build_style_bank --max_images 100
```

### Test specific method only:
```bash
python test_style_transfer.py --method histogram --source_domain iSAID --target_domain DeepGlobe
```

## Output

The script creates:
- **Visual comparisons**: Side-by-side images showing original, target reference, and all transformations
- **Quantitative metrics**: Histogram distance and color moment distance measurements
- **Style bank**: JSON file with learned domain statistics for future use

## Understanding the Results

**Lower scores = better style transfer quality**

- **Histogram Distance**: Measures how similar the color distributions are (0 = identical)
- **Color Moment Distance**: Measures similarity of mean colors and color variation

## File Structure

Results are saved to `./style_transfer_results/` with:
- `{source}_to_{target}_image_{n}.png` - Visual comparisons
- `{source}_to_{target}_metrics.png` - Bar charts of quantitative results  
- `{source}_to_{target}_metrics.json` - Detailed numerical results
- `style_bank.json` - Learned domain statistics

## Next Steps

1. **Test the script** with your datasets to see which method works best
2. **Analyze results** to understand what transformations help most
3. **Integrate the best method** into your main training pipeline
4. **Use style banks** for consistent transformations during training

The style bank approach is particularly promising as it learns robust statistics from multiple images rather than relying on a single reference image.