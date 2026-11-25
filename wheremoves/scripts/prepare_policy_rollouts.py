"""Install verified successful-policy recordings and selector maps for the first demo."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CAMERAS = ['robot0_agentview_left','robot0_agentview_right','robot0_eye_in_hand']
TITLES = {
    'CloseDoubleDoor':'Close the cabinet doors', 'CloseSingleDoor':'Close a cabinet door',
    'OpenDoubleDoor':'Open the cabinet doors', 'OpenSingleDoor':'Open a cabinet door',
    'CloseDrawer':'Close a drawer', 'OpenDrawer':'Open a drawer',
    'TurnSinkSpout':'Turn the sink spout', 'TurnOnStove':'Turn on the stove',
    'TurnOffStove':'Turn off the stove', 'CoffeeSetupMug':'Set up a coffee mug',
    'CoffeeServeMug':'Serve a coffee mug', 'CoffeePressButton':'Start the coffee machine',
    'PnPCounterToCab':'Counter to cabinet', 'PnPCabToCounter':'Cabinet to counter',
    'PnPCounterToStove':'Counter to stove', 'PnPStoveToCounter':'Stove to counter',
    'PnPCounterToSink':'Counter to sink', 'PnPSinkToCounter':'Sink to counter',
    'PnPCounterToMicrowave':'Counter to microwave', 'PnPMicrowaveToCounter':'Microwave to counter',
    'TurnOnMicrowave':'Turn on the microwave', 'TurnOffMicrowave':'Turn off the microwave',
    'TurnOnSinkFaucet':'Turn on the faucet', 'TurnOffSinkFaucet':'Turn off the faucet',
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, separators=(',', ':'), allow_nan=False))
    temporary.replace(path)


def paths_for_labels(labels, k):
    height, width = labels.shape
    paths = [''] * (k + 1)
    for y, row in enumerate(labels):
        start = 0
        for x in range(1, width + 1):
            if x == width or row[x] != row[start]:
                paths[int(row[start])] += f'M{start} {y}h{x-start}v1h{start-x}z'
                start = x
    return paths


def prepare(source):
    ranking = json.loads((source/'ranking.json').read_text())
    selected = ranking['selected']
    assert len(selected) == 8 and all(r['camera'] in [0,1] for r in selected)
    ids = list(dict.fromkeys(r['task']+'-'+r['demo'] for r in selected))
    manifests = {identity:json.loads((source/'scenes'/identity/'manifest.json').read_text()) for identity in ids}
    # Validate all source content before replacing the scene list.
    for identity, m in manifests.items():
        assert m['success'] and m['complete'] and m['episodeFrames'] == m['steps']+1
        assert m['captureSteps'] == m['steps']
        assert re.fullmatch('[a-f0-9]{64}',m['capturedOutcomeSHA256'])
        assert m['queries'] == list(range(0,m['steps'],16))
        assert len(m['views']) == len(m['queries'])*3
        assert {(r['queryStep'],r['camera']) for r in m['views']} == {(q,c) for q in m['queries'] for c in range(3)}
        for r in m['views']:
            assert digest(source/r['archive']) == r['archiveSHA256']
            assert digest(source/r['image']) == r['imageSHA256']
            with Image.open(source/r['image']) as im:
                assert im.size == (2048,2048)
            with np.load(source/r['archive']) as z:
                assert hashlib.sha256(z['rgb'].tobytes()).hexdigest() == r['sourceRGBSHA256']
                for k in ranking['ks']:
                    assert z[f'prediction_k{k}'].shape == (128,128)
                    assert set(np.unique(z[f'prediction_k{k}'])).issubset(set(range(k))|{-1})
                    assert z[f'points_k{k}'].shape == (k,2)
                    assert np.isfinite(z[f'points_k{k}']).all()
    views = []
    for identity, m in manifests.items():
        base = Path('data/policy-rollouts')/identity
        for r in m['views']:
            folder = base/f'q{r["queryStep"]:06d}'
            (ROOT/folder).mkdir(parents=True,exist_ok=True)
            image = folder/f'camera-{r["camera"]}.jpg'
            # Keep the native resolution and full chroma detail while reducing
            # the download size of the web copy. Preserve the source hash too.
            with Image.open(source/r['image']) as rendered:
                rendered.save(ROOT/image,quality=90,subsampling=0,optimize=True)
            image_hash = digest(ROOT/image)
            with np.load(source/r['archive']) as z:
                for k in ranking['ks']:
                    original = z[f'prediction_k{k}']; labels = original.copy(); labels[labels == -1] = k
                    write_json(ROOT/folder/f'k{k}-camera{r["camera"]}.json',dict(
                        k=k,camera=r['camera'],queryStep=r['queryStep'],width=128,height=128,
                        image=str(image),imageWidth=2048,imageHeight=2048,imageSHA256=image_hash,
                        sourceImageSHA256=r['imageSHA256'],
                        sourceRGBSHA256=r['sourceRGBSHA256'],sourceArchiveSHA256=r['archiveSHA256'],
                        rawPoints=z[f'points_k{k}'].tolist(),groupPaths=paths_for_labels(labels,k),
                        groupCounts=np.bincount(labels.ravel(),minlength=k+1).tolist(),
                        predictionSHA256=hashlib.sha256(original.astype('<i2').tobytes()).hexdigest()))
            views.append(dict(r,scene=identity,image=str(image),sourceImageSHA256=r['imageSHA256'],imageSHA256=image_hash))
        print(f'Installed {identity}: {len(m["queries"])} queries, {m["episodeFrames"]} recorded frames',flush=True)
    scenes = []
    task_totals = {r['task']:sum(x['task']==r['task'] for x in selected) for r in selected}
    task_counts = {}
    for rank, row in enumerate(selected,1):
        identity = row['task']+'-'+row['demo']; m=manifests[identity]
        task_counts[row['task']] = task_counts.get(row['task'],0)+1
        title = TITLES.get(row['task'], re.sub(r'(?<=[a-z])(?=[A-Z])',' ',row['task']))
        if task_totals[row['task']] > 1:
            title += f' · scene {task_counts[row["task"]]}'
        scenes.append(dict(id=row['id'],rolloutId=identity,name=title,task=row['task'],episode=row['demo'],
            sourceKind='successful_policy_rollout',success=True,rank=rank,rankedView=row['id'],
            queryStep=row['queryStep'],querySteps=m['queries'],episodeFrames=m['episodeFrames'],
            defaultCamera=row['camera'],cameras=[0,1,2],budgets=ranking['ks'],instruction=m['instruction'],
            minimumBudgetPercentile=row['minimumBudgetPercentile'],scores=row['scores'],
            queryRoot=f'data/policy-rollouts/{identity}'))
    provenance = dict(sourceKind='successful_policy_rollout',policyK=4,planningInterval=16,
        weightsSHA256='aacf0756b0ca4cff107095fde2f3ad6c6f25941dda5f2bdbe4592fa215d588f1',
        rankingSHA256=digest(source/'ranking.json'),ranking=ranking['ranking'],metric=ranking['metric'],
        cohort=ranking['cohort'],cohortCount=ranking['cohortCount'],successfulRollouts=ranking['successfulRollouts'],
        rankedViews=[r['id'] for r in selected],rankingCameras=CAMERAS[:2],companionCameras=CAMERAS,
        selectorOutputs='Recomputed on native 128px renders of recorded policy states, at the original 16-step planning times. '
            'Changing K changes selector outputs on the same successful K4 rollout; it does not change the recorded actions.',
        playback='Native 2048px renders at every recorded prediction step, 16 actions apart. No new actions or trajectory forecasts.')
    provenance['imageEncoding'] = 'Native 2048x2048 JPEG, quality 90, no chroma subsampling; source JPEG quality 95 retained outside the website.'
    source_scenes = [{k:v for k,v in m.items() if k not in {'videos','views'}} for m in manifests.values()]
    write_json(ROOT/'data/policy-rollouts.json',dict(complete=True,**provenance,scenes=source_scenes,views=views))
    write_json(ROOT/'data/groups-provenance.json',provenance)
    write_json(ROOT/'data/groups.json',scenes)
    rendering = json.loads((ROOT/'data/image-rendering.json').read_text())
    rendering['records'] = [r for r in rendering['records'] if r['kind']=='robocasa']
    write_json(ROOT/'data/image-rendering.json',rendering)
    print(f'Installed {len(scenes)} ranked views from {len(manifests)} successful policy rollouts.')


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    prepare(p.parse_args().source)
