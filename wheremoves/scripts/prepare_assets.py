"""Prepare browser-sized exports without changing model predictions.

Usage: python scripts/prepare_assets.py --downloads /path/to/Downloads
Dependencies: numpy, Pillow. Original inference arrays never enter the site.
"""
from pathlib import Path
import argparse
import base64
import hashlib
import json
import re
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, separators=(",", ":"), allow_nan=False))


def project(points, camera):
    h = np.concatenate([points, np.ones((*points.shape[:-1], 1))], axis=-1)
    result = h @ camera.T
    return result[..., :2] / result[..., 2:3]


def display_tracks(points, camera):
    """Break paths behind the camera; preserve off-screen points in front.

    A 10-micrometer near plane avoids dividing by effectively zero depth.
    Camera orientation and metric scale are fixed in fit_camera.
    """
    projected = np.round(project(points, camera), 4).tolist()
    depth = points @ camera[2, :3] + camera[2, 3]
    for i, t in np.argwhere(depth <= 1e-5):
        projected[i][t] = None
    return projected


def fit_camera(world, valid, width=128):
    """Recover the fixed pinhole camera from exact dense 3D/pixel pairs.

    The fit is checked on EVERY valid pixel, not just the fitting subset.
    """
    indices = np.flatnonzero(valid)
    subset = indices[::max(1, len(indices) // 500)]
    xyz = np.c_[world[subset], np.ones(len(subset))]
    uv = np.c_[subset % width, subset // width]
    zero = np.zeros_like(xyz)
    a = np.r_[np.c_[xyz, zero, -uv[:, :1] * xyz],
              np.c_[zero, xyz, -uv[:, 1:] * xyz]]
    _, _, vt = np.linalg.svd(a, full_matrices=False)
    camera = vt[-1].reshape(3, 4)
    camera /= np.linalg.norm(camera[2, :3])
    depth = world[indices] @ camera[2, :3] + camera[2, 3]
    if np.median(depth) < 0:
        camera *= -1
    assert (world[indices] @ camera[2, :3] + camera[2, 3] > 0).all()
    expected = np.c_[indices % width, indices // width]
    error = np.linalg.norm(project(world[indices], camera) - expected, axis=1)
    assert error.max() < 0.01, f"Camera reprojection error {error.max()}"
    return camera, float(error.max())


def allocations(capacity, sizes, k, total):
    order = sorted(range(k), key=lambda i: (-sizes[i], i))
    base, remainder = divmod(total-k, k)
    requested = np.full(k, base, dtype=int)
    requested[order[:remainder]] += 1
    allocated = np.minimum(requested, capacity)
    deficit = total-k-int(allocated.sum())
    while deficit:
        available = [i for i in order if allocated[i] < capacity[i]]
        if not available:
            break
        for i in available[:deficit]:
            allocated[i] += 1
            deficit -= 1
    repeat_order = [i for i in order if capacity[i]] or order
    for i in range(deficit):
        allocated[repeat_order[i % len(repeat_order)]] += 1
    return allocated


def prepare_robocasa(source_root):
    specs = [('hotdog', 'Pick and place', 'PnPCounterToCab', 'demo_2582', 48, 1),
             ('cabinet', 'Open a cabinet', 'OpenSingleDoor', 'demo_122', 224, 0),
             ('drawer', 'Close a drawer', 'CloseDrawer', 'demo_1438', 160, 1),
             ('stove', 'Turn on a stove', 'TurnOnStove', 'demo_1337', 16, 1)]
    evidence = []
    scenes = []
    for scene, title, task, demo, step, default_camera in specs:
      source = source_root / scene / 'frozen_predictions'
      if not source.is_dir():
        continue
      budgets = sorted(int(p.name[6:]) for p in source.glob('ours_k*') if (p/'predictions.npz').exists())
      output_dir = ROOT / 'data/robocasa' / scene
      output_dir.mkdir(parents=True, exist_ok=True)
      instruction = ''
      reference_rgb = None
      for k in budgets:
        path = source / f'ours_k{k}/predictions.npz'
        data = np.load(path, allow_pickle=False)
        metadata = json.loads(path.with_name('metadata.json').read_text())
        instruction = metadata['instruction']
        if reference_rgb is None:
            reference_rgb = data['rgb_images'].copy()
        else:
            np.testing.assert_array_equal(reference_rgb, data['rgb_images'])
        assert data['query_valid'].all(), f'Invalid query slots: {scene}, {k}'
        assert data['policy_valid'].all(), f'Invalid policy slots: {scene}, {k}'
        for c in range(3):
            dense = data['dense_pixel_features'][c].reshape(-1, 8)
            emb = data['point_embeddings'][c]
            similarities = (dense / np.linalg.norm(dense, axis=1, keepdims=True)) @ (emb / np.linalg.norm(emb, axis=1, keepdims=True)).T
            # This export orders representatives first, then background LAST.
            labels = similarities.argmax(axis=1)
            centers = data['query_flat_pixels'][c]
            valid = data['scene_valid'].reshape(3, -1)[c]
            world = data['scene_world'].reshape(3, -1, 3)[c]
            masked_labels = labels.copy()
            masked_labels[~valid] = k
            camera, projection_error = fit_camera(world, valid)
            diagnostics = metadata.get('diagnostics') or {}
            diagnostic = diagnostics.get('views', [{}]*3)[c]
            counts = np.bincount(masked_labels, minlength=k + 1)[:k]
            if 'cluster_pixel_counts' in diagnostic:
                assert counts.tolist() == diagnostic['cluster_pixel_counts'], (scene, k, c, 'group counts')
            pools = []
            for g in range(k):
                pool = np.flatnonzero(masked_labels == g)
                pool = pool[~np.isin(pool, centers)]
                ranked = pool[np.argsort(-similarities[pool, g], kind='stable')]
                pools.append(ranked.tolist())
            # Account for small groups and the source protocol's repeat rule.
            capacities = np.array([len(pool) for pool in pools])
            max_counts = np.stack([allocations(capacities, counts, k, n) for n in range(k, 257)]).max(axis=0)
            pools = [pool[:min(len(pool), int(max_counts[g]))] for g, pool in enumerate(pools)]
            # Verify all exact saved N=256 policy displacements and parents.
            # Direct K=256 exports omit the redundant identity parent mapping.
            parents = data['policy_parent_queries'][c] if 'policy_parent_queries' in data else np.arange(k)
            assert len(parents) == 256
            query = data['query_tracks_world'][c]
            policy = data['policy_tracks_world'][c]
            offsets = query - query[:, :1]
            copied = policy[:, :1] + offsets[parents]
            displacement_error = float(np.abs(copied - policy).max())
            assert displacement_error < 1e-5
            ranking_matches = 0
            for g in range(k):
                actual = data['policy_flat_pixels'][c][parents == g]
                # Export interleaves each mandatory representative with its pool.
                assert actual[0] == centers[g], (scene, k, c, g, 'center')
                if len(actual) > 1:
                    expected = np.resize(pools[g] or [int(centers[g])], len(actual)-1).tolist()
                    assert actual[1:].tolist() == expected, (scene, k, c, g, 'ranking')
                ranking_matches += len(actual)
            group_paths = [''] * (k + 1)
            grid = labels.reshape(128, 128)
            # Run-length SVG paths preserve every pixel boundary at any zoom.
            for y, row in enumerate(grid):
                start = 0
                for x in range(1, 129):
                    if x == 128 or row[x] != row[start]:
                        group_paths[int(row[start])] += f'M{start} {y}h{x-start}v1h{start-x}z'
                        start = x
            image = data['rgb_images'][c].transpose(1, 2, 0)
            Image.fromarray(image).save(output_dir / f'camera-{c}.png')
            # Candidate trajectories are preprojected from actual 3D displacements.
            candidates = []
            for g, pool in enumerate(pools):
                positions = world[pool]
                tracks = positions[:, None, :] + offsets[g][None, :, :]
                # Reuse the exact stored world tracks wherever available. The
                # export's float32 coordinate conversion differs from subtracting
                # rounded query positions by up to a few ULPs. Near the camera
                # plane, that can magnify dramatically under perspective division.
                for i, pixel in enumerate(pool):
                    matches = np.flatnonzero((parents == g) & (data['policy_flat_pixels'][c] == pixel))
                    if len(matches):
                        exact = policy[matches[0]]
                        assert np.abs(tracks[i] - exact).max() < 1e-5
                        tracks[i] = exact
                candidates.append({'pixels': pool, 'tracks': display_tracks(tracks, camera)})
            output = {
                'k': k, 'camera': c, 'width': 128, 'height': 128,
                'image': f'data/robocasa/{scene}/camera-{c}.png',
                'rawPoints': data['raw_xy'][c].tolist(),
                'queryPoints': data['query_xy'][c].tolist(),
                'groupPaths': group_paths,
                'groupCounts': np.bincount(labels, minlength=k+1).tolist(),
                'poolCounts': counts.tolist(), 'poolCapacities': capacities.tolist(), 'candidates': candidates,
                'tracks': display_tracks(query, camera),
                'savedPolicy': {
                    'points': data['policy_xy'][c].tolist(),
                    'parents': parents.tolist(),
                    'tracks': display_tracks(policy, camera),
                },
            }
            write_json(output_dir / f'k{k}-camera{c}.json', output)
            evidence.append({'scene': scene, 'k': k, 'camera': c, 'projectionMaxErrorPx': projection_error,
                             'displacementMaxError': displacement_error,
                             'savedPolicyPointsVerified': ranking_matches,
                             'sourceSHA256': hashlib.sha256(path.read_bytes()).hexdigest()})
      scenes.append({'id': scene, 'name': title, 'task': task, 'demo': demo, 'queryStep': step,
                     'defaultCamera': default_camera, 'budgets': budgets, 'instruction': instruction})
    assert scenes, 'No RoboCasa scenes found'
    write_json(ROOT / 'data/robocasa.json', scenes)
    write_json(ROOT / 'data/verification.json', evidence)


def externalize_svg_images():
    """Deduplicate embedded original images; keep vector and text elements intact."""
    image_dir = ROOT / 'assets/figures/images'
    image_dir.mkdir(exist_ok=True)
    def replace(match):
        raw = base64.b64decode(match[2])
        name = hashlib.sha256(raw).hexdigest()[:16] + '.' + match[1].replace('jpeg', 'jpg')
        (image_dir / name).write_bytes(raw)
        # Figures are inserted inline, so paths are relative to index.html.
        return 'assets/figures/images/' + name
    for path in (ROOT / 'assets/figures').glob('*.svg'):
        svg = re.sub(r'data:image/(png|jpeg);base64,([A-Za-z0-9+/=\s]+)(?=")', replace, path.read_text())
        path.write_text(svg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--downloads', type=Path, required=True)
    parser.add_argument('--robocasa-source', type=Path)
    args = parser.parse_args()
    prepare_robocasa(args.robocasa_source or args.downloads / 'wheremoves-site-exports-20261123')
    externalize_svg_images()
    print('Prepared real predictions and verified every saved 256-point output.')
