# Point Cloud Contour Generator

Generate contour lines from terrain point cloud data with Delaunay triangulation and edge filtering.

## Usage

```bash
uv run contourer data/zakazka-body.txt [options]
```

### Options

| Option | Description |
|--------|-------------|
| `--levels 495 496 ...` | Custom Z contour levels (auto-generated if omitted) |
| `--minor-interval 0.2` | Interval for minor contour lines (overrides --levels) |
| `--major-interval 1.0` | Interval for major contour lines (visualization only, defaults to 5× minor) |
| `--max-distance 5.0` | Max triangle edge length filter (default: 1.5× median) |
| `--output FILE` | Output file path (default: `./data/out/contour_lines.txt`) |
| `--format {txt,geojson}` | Export format (default: txt) |
| `--plot map.pdf` | Save visualization to vector PDF file |
| `--show-points` | Show original data points on the visualization |

### Examples

```bash
# Auto levels with visualization
uv run contourer data/zakazka-body.txt --plot data/out/contour_intervals.pdf

# Major and minor contours at 0.2 and 1 meter
uv run contourer data/zakazka-body.txt --minor-interval 0.2 --major-interval 1 --plot data/out/contour_intervals.pdf

# Show original data points on the map
uv run contourer data/zakazka-body.txt --plot data/out/map.pdf --show-points

# Custom levels with GeoJSON export
uv run contourer data/zakazka-body.txt --levels 495 496 497 498 499 500 --format geojson
```

## Input Format

Space-separated: `ID X Y Z` (columns 2-4 used)

## Output Formats

- **txt**: `z_value: x1,y1; x2,y2; ...` per segment
- **geojson**: FeatureCollection with LineString geometries
