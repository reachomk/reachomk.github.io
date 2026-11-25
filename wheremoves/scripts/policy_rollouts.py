"""Rank recorded successful policy rollouts and export selector-only website assets.

Run on the original RoboCasa host. Never loads the action policy or trajectory
predictor. Simulator states are restored, not simulated. Selector outputs are
recomputed on RGB rendered from the recorded states.
"""
import os
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time
import pyexpat
import mujoco
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from scipy.stats import rankdata

CAMERAS = ['robot0_agentview_left', 'robot0_agentview_right', 'robot0_eye_in_hand']
KS = [4, 8, 16, 32, 64, 128, 256]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False))
    tmp.replace(path)


class RecordedScene:
    def __init__(self, folder):
        from robocasa_gt_tracks.clustering import _set_saved_state
        self.restore = _set_saved_state
        self.folder = Path(folder)
        self.summary = json.loads((self.folder/'summary.json').read_text())
        assert self.summary['success'] is True
        self.states = np.load(self.folder/'states_actions.npz')['states']
        assert len(self.states) == self.summary['steps'] + 1
        self.xml = gzip.open(self.folder/'model.xml.gz', 'rt').read()
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.model.vis.global_.offwidth = max(2048, self.model.vis.global_.offwidth)
        self.model.vis.global_.offheight = max(2048, self.model.vis.global_.offheight)
        self.data = mujoco.MjData(self.model)
        self.option = mujoco.MjvOption()
        self.option.geomgroup[:] = 0
        self.option.geomgroup[1] = 1
        self.option.sitegroup[:] = 0
        self.renderers = {}

    def set_step(self, step):
        self.restore(self.model, self.data, self.states[step])

    def renderer(self, size):
        if size not in self.renderers:
            self.renderers[size] = mujoco.Renderer(self.model, height=size, width=size)
        return self.renderers[size]

    def rgb(self, camera, size=128):
        renderer = self.renderer(size)
        renderer.disable_depth_rendering()
        renderer.disable_segmentation_rendering()
        renderer.update_scene(self.data, camera=CAMERAS[camera], scene_option=self.option)
        return renderer.render().copy()

    def motion(self, camera, step):
        from robocasa_gt_tracks.clustering import _sample_query_pixels
        self.set_step(step)
        renderer = self.renderer(128)
        renderer.enable_depth_rendering()
        cid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERAS[camera])
        f = 64 / np.tan(np.deg2rad(self.model.cam_fovy[cid])/2)
        intrinsic = np.array([[f,0,64],[0,f,64],[0,0,1]])
        local, gids, uv = _sample_query_pixels(self.model, self.data, renderer,
            self.option, CAMERAS[camera], cid, intrinsic)
        unique, inverse = np.unique(gids, return_inverse=True)
        positions, rotations = [], []
        for t in range(step, step+33):
            self.set_step(t)
            positions.append(self.data.geom_xpos[unique].copy())
            rotations.append(self.data.geom_xmat[unique].reshape(-1,3,3).copy())
        world = np.einsum('tnij,nj->tni', np.array(rotations)[:,inverse], local) + np.array(positions)[:,inverse]
        motion = np.ascontiguousarray((world-world[:1]).transpose(1,0,2), dtype=np.float32)
        moving = np.linalg.norm(motion[:,-1], axis=-1) > .001
        self.set_step(step)
        return motion[moving], uv[moving], uv

    def close(self):
        for r in self.renderers.values():
            r.close()


def selector(root):
    from reproduction.flow_prediction.rollout import FrozenSelector
    weights = root/'outputs/flow_prediction/robocasa_joint_v1/selector_weights.pt'
    assert digest(weights) == 'aacf0756b0ca4cff107095fde2f3ad6c6f25941dda5f2bdbe4592fa215d588f1'
    model = FrozenSelector().cuda().eval().requires_grad_(False)
    model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True, mmap=True), strict=True)
    return model


def predict(model, rgb):
    image = torch.from_numpy(rgb).cuda().permute(2,0,1).float()/255
    mean = image.new_tensor([.485,.456,.406])[:,None,None]
    std = image.new_tensor([.229,.224,.225])[:,None,None]
    result = {}
    for k in KS:
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            y,x,dense,embeddings = model(((image-mean)/std)[None], k)
        similarity = F.normalize(dense[0].float(), dim=-1) @ F.normalize(embeddings[0].float(), dim=-1).T
        labels = similarity.argmax(-1).cpu().numpy().astype(np.int16)
        labels[labels == k] = -1
        result[f'prediction_k{k}'] = labels
        result[f'points_k{k}'] = np.stack([x[0].float().cpu().numpy()*127, y[0].float().cpu().numpy()*127], -1)
    return result


def score(gt, prediction, k):
    valid = gt != -2
    g, p = gt[valid].astype(int), prediction[valid].astype(int)
    g[g < 0] = k
    p[p < 0] = k
    table = np.bincount(g*(k+1)+p, minlength=(k+1)**2).reshape(k+1,k+1)
    intersection = table[:k,:k]
    union = table[:k].sum(1)[:,None] + table[:,:k].sum(0)[None] - intersection
    iou = intersection / np.maximum(union,1)
    rows, cols = linear_sum_assignment(iou, maximize=True)
    return dict(matched_miou=float(iou[rows,cols].mean()),
        foreground_iou=float(intersection.sum()/max(1,len(g)-table[k,k])),
        moving_pixels=int(table[:k].sum()))


def inventory(args):
    root = args.root/'outputs/validation_k4_all1200_20261108/episodes'
    successful, candidates = [], []
    for path in sorted(root.glob('*/*/summary.json')):
        row = json.loads(path.read_text())
        if row.get('success') is not True:
            continue
        successful.append(row)
        queries = [q for q in row['queries'] if q['actual_horizon_steps'] == 32]
        def motion(q):
            return max((min(v.get('object_actual_peak_cm') or 0, v.get('robot_actual_peak_cm') or 0)
                for v in q['views']), default=0)
        if not queries:
            continue
        query = max(queries, key=motion)
        if motion(query) < 1:
            continue
        candidates.append(dict(task=row['task'],demo=row['demo'],sourceDirectory=str(path.parent.resolve()),
            queryStep=query['step'],queryIndex=query['index'],steps=row['steps'],instruction=row['instruction'],
            objectRobotMotionCm=motion(query)))
    write_json(args.output/'candidates.json', candidates)
    write_json(args.output/'inventory.json', dict(successfulRollouts=len(successful),
        planningQueries=sum(len(r['queries']) for r in successful),candidateRollouts=len(candidates)))


def rank(args):
    from robocasa_gt_tracks.clustering import ClusterConfig, clara_kmedoids
    records = json.loads(args.candidates.read_text())
    model = selector(args.root)
    started = time.time()
    jobs = records[args.worker::args.workers]
    if args.limit:
        jobs = jobs[:args.limit]
    for i, record in enumerate(jobs):
        identity = record['task']+'-'+record['demo']
        out = args.output/'ranking'/identity
        out.mkdir(parents=True, exist_ok=True)
        if all((out/f'camera-{c}.json').exists() for c in range(2)):
            continue
        scene = RecordedScene(record['sourceDirectory'])
        try:
            for camera in range(2):
                path = out/f'camera-{camera}.json'
                if path.exists():
                    continue
                step = record['queryStep']
                scene.set_step(step)
                rgb = scene.rgb(camera)
                reference = np.array(Image.open(scene.folder/f'q{record["queryIndex"]:03d}_c{camera}.jpg').convert('RGB'))
                mae = float(np.abs(rgb.astype(float)-reference).mean())
                if mae >= 10:
                    write_json(path, dict(record, camera=camera,
                        excluded='Rendered query differs from archived query RGB by at least 10/255 mean absolute error', sourceRGBMAE=mae))
                    continue
                motion, pixels, known = scene.motion(camera, step)
                if len(motion) < max(KS):
                    write_json(path, dict(record, camera=camera, excluded='Fewer than 256 moving pixels', movingPixels=len(motion)))
                    continue
                arrays = dict(rgb=rgb, **predict(model, rgb))
                scores = {}
                for k in KS:
                    seed = int.from_bytes(hashlib.sha256(f'{identity}/{camera}/{step}'.encode()).digest()[:8], 'little')
                    medoids, labels, distances, costs = clara_kmedoids(motion, pixels, np.ones(len(motion),np.float32),
                        ClusterConfig(k=k, clara_sample_size=2048, clara_repeats=3),
                        np.random.default_rng(seed), backend='torch', device='cuda')
                    gt = np.full((128,128), -2, np.int16)
                    gt[known[:,1],known[:,0]] = -1
                    gt[pixels[:,1],pixels[:,0]] = labels
                    arrays[f'gt_k{k}'] = gt
                    arrays[f'gt_points_k{k}'] = pixels[medoids]
                    scores[str(k)] = score(gt, arrays[f'prediction_k{k}'], k)
                archive = out/f'camera-{camera}.npz'
                np.savez_compressed(archive, **arrays)
                write_json(path, dict(record, id=f'{identity}-c{camera}', camera=camera,
                    sourceRGBMAE=mae, archive=str(archive.relative_to(args.output)),
                    archiveSHA256=digest(archive), scores=scores))
                print(json.dumps(dict(worker=args.worker,done=i+1,total=len(jobs),id=identity,camera=camera,
                    seconds=round(time.time()-started,1),scores={k:round(s['matched_miou'],3) for k,s in scores.items()})), flush=True)
        finally:
            scene.close()


def finalize(args):
    records = [json.loads(p.read_text()) for p in (args.output/'ranking').glob('*/camera-*.json')]
    expected = len(json.loads(args.candidates.read_text())) * 2
    assert len(records) == expected, (len(records), expected)
    # The source training protocol rejects queries whose K clusters are not
    # all populated. Apply that rule here as well, across every budget.
    for record in records:
        if 'scores' not in record:
            continue
        with np.load(args.output/record['archive']) as arrays:
            invalid = [k for k in KS if len(np.unique(arrays[f'gt_k{k}'][arrays[f'gt_k{k}']>=0])) != k]
        if invalid:
            record['excluded'] = f'Ground-truth clustering has empty clusters at K={invalid}'
    valid = [r for r in records if 'scores' in r and 'excluded' not in r]
    matrix = np.array([[r['scores'][str(k)]['matched_miou'] for k in KS] for r in valid])
    percentiles = np.stack([(rankdata(matrix[:,j])-1)/(len(valid)-1) for j in range(len(KS))],1)
    for r, row in zip(valid, percentiles):
        r['minimumBudgetPercentile'] = float(row.min())
        r['meanBudgetPercentile'] = float(row.mean())
    valid.sort(key=lambda r:(r['minimumBudgetPercentile'],r['meanBudgetPercentile']), reverse=True)
    write_json(args.output/'ranking.json', dict(ks=KS, cohortCount=len(valid),
        successfulRollouts=632, candidateRollouts=expected//2,
        cohort='Both external cameras at one informative recorded query per eligible successful K4 policy rollout. '
            'Eligible queries have a full 32-step future and at least 1 cm of observed object and robot motion; '
            'the query maximizing the smaller of those two motions is used. Views with fewer than 256 moving pixels '
            'or rendered/archived RGB mean absolute error of at least 10/255 are excluded. '
            'Every ground-truth cluster must be populated at all seven budgets.',
        metric='Hungarian-matched mean IoU against CLARA ground-truth motion clusters, independently for K=4,8,16,32,64,128,256. '
            'Ground truth uses recorded material motion over 33 states, with endpoint motion above 1 mm. '
            'Static pixels count as false positives, unknown pixels are ignored; background cannot match a moving cluster.',
        ranking='Descending minimum within-budget percentile, then mean percentile.',
        excluded=[r for r in records if 'excluded' in r], selected=valid[:8], records=valid))
    print(json.dumps(valid[:8], indent=2), flush=True)


def export(args):
    import imageio.v2 as imageio
    ranking = json.loads((args.output/'ranking.json').read_text())
    unique = {r['task']+'-'+r['demo']:r for r in ranking['selected']}
    jobs = list(unique.items())[args.worker::args.workers]
    model = selector(args.root)
    for identity, record in jobs:
        out = args.output/'scenes'/identity
        out.mkdir(parents=True, exist_ok=True)
        if (out/'manifest.json').exists():
            continue
        scene = RecordedScene(record['sourceDirectory'])
        # Match the recording to its own captured-run result. The older canonical
        # benchmark run can have a different length despite sharing initialization.
        canonical = json.loads(Path(scene.summary['canonical_outcome']).read_text())
        assert canonical['complete'] and canonical['success']
        captured = json.loads(Path(scene.summary['replay_outcome']).read_text())
        assert captured['complete'] and captured['success']
        assert captured['steps'] == scene.summary['steps']
        queries = [q['step'] for q in scene.summary['queries']]
        assert queries == list(range(0, scene.summary['steps'], 16))
        views, videos, writers = [], [], []
        started = time.time()
        try:
            for camera in range(3):
                path = out/f'camera-{camera}.mp4'
                writers.append(imageio.get_writer(path, fps=20, codec='libx264', macro_block_size=16,
                    ffmpeg_params=['-crf','20','-preset','fast','-pix_fmt','yuv420p','-g','16','-keyint_min','16',
                        '-sc_threshold','0','-movflags','+faststart','-threads','2']))
            for step in range(len(scene.states)):
                scene.set_step(step)
                query = step in queries
                folder = out/f'q{step:06d}'
                if query:
                    folder.mkdir(exist_ok=True)
                for camera in range(3):
                    high = scene.rgb(camera, 2048)
                    writers[camera].append_data(high)
                    if not query:
                        continue
                    rgb = scene.rgb(camera)
                    ranked = args.output/'ranking'/identity/f'camera-{camera}.json'
                    reused = False
                    if ranked.exists():
                        metadata = json.loads(ranked.read_text())
                        reused = metadata.get('queryStep') == step and 'archive' in metadata
                    if reused:
                        with np.load(args.output/metadata['archive']) as data:
                            np.testing.assert_array_equal(data['rgb'], rgb)
                            arrays = {k:data[k].copy() for k in data.files if k.startswith(('prediction_', 'points_'))}
                    else:
                        arrays = predict(model, rgb)
                    archive = folder/f'camera-{camera}.npz'
                    np.savez_compressed(archive, rgb=rgb, **arrays)
                    image = folder/f'camera-{camera}.jpg'
                    Image.fromarray(high).save(image, quality=95, subsampling=0, optimize=True)
                    views.append(dict(queryStep=step,camera=camera,width=2048,height=2048,
                        archive=str(archive.relative_to(args.output)),archiveSHA256=digest(archive),
                        image=str(image.relative_to(args.output)),imageSHA256=digest(image),
                        sourceRGBSHA256=hashlib.sha256(rgb.tobytes()).hexdigest(),
                        stateSHA256=hashlib.sha256(scene.states[step].tobytes()).hexdigest(),
                        reusedRankedPrediction=reused))
                if step % 32 == 0:
                    print(json.dumps(dict(id=identity,step=step,total=len(scene.states),seconds=round(time.time()-started,1))),flush=True)
            for w in writers:
                w.close()
            writers = []
            for camera in range(3):
                path = out/f'camera-{camera}.mp4'
                reader = imageio.get_reader(path)
                assert reader.count_frames() == len(scene.states)
                assert reader.get_meta_data()['size'] == (2048,2048)
                reader.close()
                videos.append(dict(camera=camera,path=str(path.relative_to(args.output)),sha256=digest(path),
                    frames=len(scene.states),fps=20,width=2048,height=2048,bytes=path.stat().st_size))
            write_json(out/'manifest.json',dict(id=identity,task=record['task'],demo=record['demo'],
                instruction=scene.summary['instruction'],success=True,steps=scene.summary['steps'],
                source=record['sourceDirectory'],queries=queries,episodeFrames=len(scene.states),
                sourceSummarySHA256=digest(scene.folder/'summary.json'),
                canonicalOutcomeSHA256=digest(scene.summary['canonical_outcome']),
                canonicalSteps=canonical['steps'],captureSteps=captured['steps'],
                capturedOutcomeSHA256=digest(scene.summary['replay_outcome']),
                statesSHA256=digest(scene.folder/'states_actions.npz'),xmlSHA256=hashlib.sha256(scene.xml.encode()).hexdigest(),
                views=views,videos=videos,complete=True))
            print('COMPLETE '+identity, flush=True)
        finally:
            for w in writers:
                w.close()
            scene.close()


def verify_exports(args):
    ranking = json.loads((args.output/'ranking.json').read_text())
    ids = list(dict.fromkeys(r['task']+'-'+r['demo'] for r in ranking['selected']))
    reports = []
    for identity in ids:
        path = args.output/'scenes'/identity/'manifest.json'
        m = json.loads(path.read_text())
        summary = json.loads((Path(m['source'])/'summary.json').read_text())
        captured = json.loads(Path(summary['replay_outcome']).read_text())
        canonical = json.loads(Path(summary['canonical_outcome']).read_text())
        assert captured['complete'] and captured['success'] and canonical['success']
        assert captured['steps'] == summary['steps'] == m['steps']
        assert m['episodeFrames'] == m['steps']+1
        assert m['queries'] == list(range(0,m['steps'],16))
        m.update(captureSteps=captured['steps'],canonicalSteps=canonical['steps'],
            capturedOutcomeSHA256=digest(summary['replay_outcome']))
        write_json(path,m)
        reports.append(dict(id=identity,steps=m['steps'],queries=len(m['queries']),videos=len(m['videos']),
            videoBytes=sum(v['bytes'] for v in m['videos']),captureSuccess=True,canonicalSuccess=True))
    write_json(args.output/'verification.json', dict(complete=True,records=reports))
    print(json.dumps(reports,indent=2),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['inventory', 'rank', 'finalize', 'export', 'verify_exports'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--candidates', type=Path)
    parser.add_argument('--worker', type=int, default=0)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    sys.path[:0] = [str(args.root), str(args.root/'src'), str(args.root/'point_selector')]
    torch.set_num_threads(2)
    globals()[args.mode](args)


if __name__ == '__main__':
    main()
