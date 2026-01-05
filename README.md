# Point Cloud Contour Generator

Generate contour lines from terrain point cloud data with Delaunay triangulation and edge filtering.

## Usage

```bash
uv run contourer data/zakazka-body.txt [options]
```

### Options

| Option | Description |
|--------|-------------|
| `--levels 495 496 ...` | Custom Z contour levels (default: 50 auto-generated) |
| `--max-distance 5.0` | Max triangle edge length filter (default: 3× median) |
| `--output FILE` | Output file path (default: `contour_lines.txt`) |
| `--format {txt,geojson}` | Export format (default: txt) |
| `--plot map.png` | Save visualization image |

### Examples

```bash
# Auto levels with visualization
uv run contourer data/zakazka-body.txt --plot map.png

# Custom levels with GeoJSON export
uv run contourer data/zakazka-body.txt --levels 495 496 497 498 499 500 --format geojson
```

## Input Format

Space-separated: `ID X Y Z` (columns 2-4 used)

## Output Formats

- **txt**: `z_value: x1,y1; x2,y2; ...` per segment
- **geojson**: FeatureCollection with LineString geometries
