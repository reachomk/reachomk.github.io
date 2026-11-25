"""Re-render the website's saved RoboCasa observations at native high resolution.

Run in the source repository's flow-rollout environment. No simulation steps,
model inference, camera changes, or prediction changes are performed.
"""
import os
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('MUJOCO_EGL_DEVICE_ID', '0')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image
import mujoco

CAMERAS = ['robot0_agentview_left', 'robot0_agentview_right', 'robot0_eye_in_hand']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--size', type=int, default=2048)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--kind', choices=['robocasa'], default='robocasa')
    args = parser.parse_args()
    root = args.root
    sys.path.insert(0, str(root/'src'))
    from robocasa_gt_tracks.clustering import _set_saved_state
    jobs = []

    cfg = json.loads((root/'outputs/flow_prediction/robocasa_joint_v1/experiments/full_dense_scene_4x_accum4_v1/six_settings_100000_full50_v1/config.json').read_text())
    for scene in ['hotdog', 'cabinet', 'drawer', 'stove']:
        package = root/'outputs/wheremoves_site_exports_20261123'/scene
        meta = json.loads((package/'g003.json').read_text())
        task, demo = meta.get('task', 'PnPCounterToCab'), meta.get('demo', 'demo_2582')
        model_path = Path(cfg['initial_scene_cache'])/'validation'/task/demo/'compiled_model.xml.gz'
        with np.load(package/'frozen_predictions/ours_k4/predictions.npz') as data:
            state = data['flat_state'].copy()
            references = list(data['rgb_images'].transpose(0, 2, 3, 1).copy())
        jobs.append(dict(kind='robocasa', id=scene, xml=gzip.open(model_path, 'rt').read(),
            state=state, references=references, source=str(package), query=meta['policy_step']))

    evidence = []
    for job in jobs[:args.limit]:
        model = mujoco.MjModel.from_xml_string(job['xml'])
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, args.size)
        model.vis.global_.offheight = max(model.vis.global_.offheight, args.size)
        data = mujoco.MjData(model)
        _set_saved_state(model, data, job['state'])
        option = mujoco.MjvOption()
        option.geomgroup[:] = 0
        option.geomgroup[1] = 1
        option.sitegroup[:] = 0
        folder = args.output/job['kind']/job['id']
        folder.mkdir(parents=True, exist_ok=True)
        low_renderer = mujoco.Renderer(model, height=128, width=128)
        high_renderer = None
        try:
            for c, camera in enumerate(CAMERAS):
                low_renderer.update_scene(data, camera=camera, scene_option=option)
                low = low_renderer.render().copy()
                camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
                ref = job['references'][c]
                mae = float(np.abs(low.astype(float)-ref).mean())
                flipped_mae = float(np.abs(low[::-1].astype(float)-ref).mean())
                Image.fromarray(low).save(folder/f'check-{c}-render.png')
                Image.fromarray(ref).save(folder/f'check-{c}-source.png')
                print(job['kind'], job['id'], camera, '128px MAE', round(mae, 4), 'flipped', round(flipped_mae, 4), flush=True)
                # A wrong pose, camera, or orientation must stop export.
                assert mae < flipped_mae, (job['id'], camera, mae, flipped_mae)
                assert mae < 10, (job['id'], camera, mae)
                if high_renderer is None:
                    high_renderer = mujoco.Renderer(model, height=args.size, width=args.size)
                high_renderer.update_scene(data, camera=camera, scene_option=option)
                high = high_renderer.render().copy()
                path = folder/f'camera-{c}.jpg'
                Image.fromarray(high).save(path, quality=95, subsampling=0, optimize=True)
                evidence.append(dict(kind=job['kind'], scene=job['id'], camera=c, cameraName=camera,
                    width=args.size, height=args.size, referenceResolution=[128, 128],
                    referenceMAE=mae, flippedMAE=flipped_mae, queryStep=job['query'],
                    appearanceMatchesReference=mae < 10,
                    cameraPosition=data.cam_xpos[camera_id].tolist(),
                    cameraRotation=data.cam_xmat[camera_id].tolist(),
                    cameraFovy=float(model.cam_fovy[camera_id]),
                    source=job['source'], xmlSHA256=hashlib.sha256(job['xml'].encode()).hexdigest(),
                    stateSHA256=hashlib.sha256(job['state'].tobytes()).hexdigest(),
                    imageSHA256=hashlib.sha256(path.read_bytes()).hexdigest()))
        finally:
            low_renderer.close()
            if high_renderer is not None:
                high_renderer.close()
        (args.output/'manifest.json').write_text(json.dumps(evidence, indent=2))
        print('Completed', job['id'], flush=True)


if __name__ == '__main__':
    main()
