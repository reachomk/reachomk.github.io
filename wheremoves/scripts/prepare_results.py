"""Join the original Figure 5 FLOP and latency reports without interpolation."""
from pathlib import Path
import argparse
import hashlib
import json
import pymupdf
import math

ROOT = Path(__file__).resolve().parents[1]
METHODS = [
    ('ours', 'WhereMoves', '#4B62B2', 'Ours', 'direct'),
    ('random', 'Random', '#C74C44', 'Random', 'direct'),
    ('grid', 'Grid', '#B87520', 'Uniform', 'direct'),
    ('dinov2', 'DINOv2', '#9462AD', 'DINOv2', 'direct'),
    ('dinov3', 'DINOv3', '#648747', 'DINOv3', 'direct'),
    ('random2048', 'Random 2,048 → 256', '#008C95', 'Overgeneration', 'random'),
    ('grid2048', 'Grid 2,048 → 256', '#B52D6A', 'Overgeneration', 'uniform'),
    ('sam3', 'SAM 3', '#8B2E1A', 'SAM 3', 'mask_grid'),
]


def main(source):
    paths = [source / 'robocasa_k_flops_total_circle_100k_selection_compute.json',
             source / 'robocasa_k_latency_3090_100k_selection_compute.json']
    flops, timing = [json.loads(p.read_text()) for p in paths]
    key = lambda row: (row['family'], row['variant'], row['generated_k'])
    timings = {key(row): row for row in timing['rows']}
    generation = {row['generated_k']: row for row in flops['rows'] if row['family'] == 'Ours'}
    output = []
    for row in flops['rows'] + flops['marker_only_rows']:
        method = next(m for m in METHODS if (m[3], m[4]) == key(row)[:2])
        k = row['generated_k']
        sam = method[0] == 'sam3'
        gen = generation[k] if sam else row
        time = timings[key(gen)]
        if not sam:
            assert row['successes'] == time['successes']
            assert row['episodes'] == time['episodes'] == 1200
            assert row['source_sha256'] == time['source_sha256']
        # Author confirmed on 2026-11-23: retain the supplied PDF's SAM 3
        # selection timing instead of the newer server measurement. SAM neural
        # selection is independent of requested K; generation remains K-specific.
        published_selection_ms = {('dinov2', 8): 87.7, ('dinov2', 256): 193.3,
                                  ('dinov3', 8): 66.8, ('dinov3', 256): 181.4}
        selection_ms = 2241.1 if sam else published_selection_ms.get((method[0], k), time['selection_ms'])
        item = dict(id=f'{method[0]}-{k}', method=method[0], k=k,
                    retainedK=row['retained_k'], successes=row['successes'], episodes=row['episodes'],
                    success=row['accuracy_percent'], generationGflops=gen['generation_gflops'],
                    selectionGflops=row['selection_gflops'], generationMs=time['generation_ms'],
                    selectionMs=selection_ms, accuracySourceSHA256=row['source_sha256'])
        item['totalGflops'] = item['generationGflops'] + item['selectionGflops']
        item['totalMs'] = item['generationMs'] + selection_ms
        output.append(item)
    assert len(output) == 44
    # Calibrate invisible hit targets from the supplied vector PDF itself.
    pdf_path = ROOT / 'assets/figures/efficiency.pdf'
    page = pymupdf.open(pdf_path)[0]
    horizontal = []
    for drawing in page.get_drawings():
        if not drawing['color'] or min(drawing['color']) < .9:
            continue
        for item in drawing['items']:
            if item[0] == 'l' and abs(item[1].y-item[2].y) < .001 and item[2].x-item[1].x > 180:
                horizontal.append(item[1].y)
    assert len(horizontal) == 7
    words = page.get_text('words')
    tick = lambda text: next(w for w in words if w[4] == text and 139 < w[1] < 141)
    a, b = tick('4'), tick('256')
    x4, x256 = (a[0]+a[2])/2, (b[0]+b[2])/2
    axis = dict(width=page.rect.width, height=page.rect.height, x4=x4,
                xPerDoubling=(x256-x4)/6, y0=max(horizontal), yPerPercent=(max(horizontal)-min(horizontal))/60)
    shapes = [d for d in page.get_drawings() if d['fill'] and 0 < d['rect'].width < 80 and 0 < d['rect'].height < 80]
    for row in output:
        x = x4 + math.log2(row['k']/4)*axis['xPerDoubling']
        y = axis['y0'] - row['success']*axis['yPerPercent']
        if row['method'] == 'sam3':
            row['hitRadius'] = 3
        else:
            matches = [s for s in shapes if math.hypot((s['rect'].x0+s['rect'].x1)/2-x, (s['rect'].y0+s['rect'].y1)/2-y) < .02]
            assert matches, f'No PDF marker at {row["id"]}'
            row['hitRadius'] = max((s['rect'].width+s['rect'].height)/4 for s in matches)
    data = dict(methods=[dict(id=m[0], name=m[1], color=m[2]) for m in METHODS], rows=output, plotCoordinates=axis,
                figureSHA256=hashlib.sha256(pdf_path.read_bytes()).hexdigest(),
                referenceSuccess=flops['reference_line']['accuracy_percent'],
                sources=[dict(name=p.name, sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths],
                generationScope=flops['flops_scope'], selectionScope=flops['selector_scope'],
                samTimingSource='website-cr.pdf, Table 4: selection 2241.1 ms; author confirmed 2026-11-23',
                publishedTimingOverrides='DINOv2/DINOv3 selection times at K=8 and 256 follow PDF Table 4; other K use the source benchmark report.',
                totals='Calculated from unrounded source stage values; SAM selection time uses the paper value.')
    (ROOT / 'data/efficiency.json').write_text(json.dumps(data, separators=(',', ':'), allow_nan=False))
    print(f'Prepared {len(output)} measured plot points from the original reports.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    main(parser.parse_args().source)
