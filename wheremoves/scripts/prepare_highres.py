"""Install verified native renders while preserving all prediction coordinates."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def prepare(source):
    records = json.loads((source/'manifest.json').read_text())
    expected = {('robocasa', scene['id'], c)
                for scene in json.loads((ROOT/'data/robocasa.json').read_text()) for c in range(3)}
    assert {(r['kind'], r['scene'], r['camera']) for r in records} == expected
    assert len(records) == len(expected)
    # Validate the complete set before replacing any site files.
    for row in records:
        path = source/row['kind']/row['scene']/f'camera-{row["camera"]}.jpg'
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row['imageSHA256']
        assert Image.open(path).size == (2048, 2048)
        assert row['referenceMAE'] < row['flippedMAE']
        assert row['referenceMAE'] < 10
    for row in records:
        directory = Path(row['kind'])/row['scene']
        filename = f'camera-{row["camera"]}.jpg'
        target = ROOT/'data'/directory
        shutil.copy2(source/directory/filename, target/filename)
        for path in target.glob(f'k*-camera{row["camera"]}.json'):
            data = json.loads(path.read_text())
            data['image'] = f'data/{directory}/{filename}'
            data['imageWidth'] = row['width']
            data['imageHeight'] = row['height']
            # width/height remain the original model-coordinate grid.
            assert data['width'] == data['height'] == 128
            path.write_text(json.dumps(data, separators=(',', ':'), allow_nan=False))
        old_image = target/f'camera-{row["camera"]}.png'
        if old_image.exists():
            old_image.unlink()
    previous_file = ROOT/'data/image-rendering.json'
    if previous_file.exists():
        previous = json.loads(previous_file.read_text())['records']
        records += [r for r in previous if (r['kind'], r['scene'], r['camera']) not in expected]
    previous_file.write_text(json.dumps(dict(
        method='Native MuJoCo RGB renders from the exact saved state and camera; no simulation steps or model re-inference.',
        resolution=[2048, 2048], encoding='JPEG quality 95, no chroma subsampling',
        predictionResolution=[128, 128], records=records), separators=(',', ':')))
    print(f'Installed {len(expected)} native 2048×2048 camera renders; prediction coordinates and masks unchanged.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--kind', choices=['robocasa'], default='robocasa')
    args = parser.parse_args()
    prepare(args.source)
