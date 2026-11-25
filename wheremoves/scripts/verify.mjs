import assert from 'node:assert/strict';
import fs from 'node:fs';
import { createHash } from 'node:crypto';
import { upsample, BUDGETS } from '../motion.js';

const root = new URL('../', import.meta.url);
const json = name => JSON.parse(fs.readFileSync(new URL(name, root), 'utf8'));
const groupScenes = json('data/groups.json');
const groupProvenance = json('data/groups-provenance.json');
const imageRendering = json('data/image-rendering.json');
assert.equal(imageRendering.records.length, 12);
assert.deepEqual(imageRendering.resolution, [2048, 2048]);
for (const row of imageRendering.records) {
  assert.equal(row.width, 2048); assert.equal(row.height, 2048);
  const imagePath = new URL(`data/${row.kind}/${row.scene}/camera-${row.camera}.jpg`, root);
  assert.equal(createHash('sha256').update(fs.readFileSync(imagePath)).digest('hex'), row.imageSHA256);
  assert.equal(row.kind, 'robocasa');
}
assert.equal(groupScenes.length, 8);
assert.deepEqual(groupProvenance.rankingCameras, ['robot0_agentview_left', 'robot0_agentview_right']);
assert.deepEqual(groupScenes.map(s => s.rankedView), groupProvenance.rankedViews);
function verifyGroupMap(data) {
  const { k } = data;
  // Decode every SVG run and compare it with the hash of the untouched
  // source prediction. This catches shifted, omitted, or changed pixels.
  const labels = new Int16Array(data.width * data.height).fill(-2);
  const counts = Array(k + 1).fill(0);
  data.groupPaths.forEach((path, group) => {
    let consumed = 0;
    for (const match of path.matchAll(/M(\d+) (\d+)h(\d+)v1h(-\d+)z/g)) {
      assert.equal(match.index, consumed); consumed += match[0].length;
      const [, sx, sy, length, backward] = match.map(Number);
      assert.equal(length, -backward);
      assert.ok(sx >= 0 && sy >= 0 && sx + length <= data.width && sy < data.height);
      for (let x = sx; x < sx + length; x++) {
        const i = sy * data.width + x;
        assert.equal(labels[i], -2, 'Each source pixel appears once');
        labels[i] = group === k ? -1 : group;
        counts[group]++;
      }
    }
    assert.equal(consumed, path.length);
  });
  assert.ok(labels.every(label => label !== -2));
  assert.deepEqual(counts, data.groupCounts);
  const bytes = Buffer.alloc(labels.length * 2);
  labels.forEach((label, i) => bytes.writeInt16LE(label, i * 2));
  assert.equal(createHash('sha256').update(bytes).digest('hex'), data.predictionSHA256);
}
let groupPredictions = 0;
for (const [index, scene] of groupScenes.entries()) {
  assert.equal(scene.rank, index + 1);
  assert.ok([0, 1].includes(scene.defaultCamera), 'Wrist cameras never determine the ranking');
  assert.equal(Number(scene.rankedView.split('-c').at(-1)), scene.defaultCamera);
  assert.equal(scene.sourceKind, 'successful_policy_rollout');
  assert.equal(scene.success, true);
  assert.deepEqual(scene.budgets, BUDGETS);
  assert.deepEqual(scene.cameras, [0, 1, 2]);
  if (index) assert.ok(groupScenes[index-1].minimumBudgetPercentile >= scene.minimumBudgetPercentile);
  for (const k of scene.budgets) for (const camera of scene.cameras) {
    const data = json(`${scene.queryRoot}/q${String(scene.queryStep).padStart(6,'0')}/k${k}-camera${camera}.json`);
    assert.equal(data.k, k); assert.equal(data.camera, camera);
    assert.equal(data.rawPoints.length, k);
    assert.ok(data.rawPoints.every(xy => xy.length === 2 && xy.every(Number.isFinite)));
    assert.equal(data.groupPaths.length, k + 1);
    assert.equal(data.groupCounts.length, k + 1);
    assert.ok(fs.existsSync(new URL(data.image, root)));
    assert.deepEqual([data.imageWidth, data.imageHeight], [2048, 2048]);
    verifyGroupMap(data);
    groupPredictions++;
  }
}
// Every saved source query must be represented for every camera and K.
const rolloutManifest = json('data/policy-rollouts.json');
assert.equal(rolloutManifest.complete, true);
assert.equal(rolloutManifest.weightsSHA256, groupProvenance.weightsSHA256);
assert.equal(rolloutManifest.rankingSHA256, groupProvenance.rankingSHA256);
const rolloutViews = new Map(rolloutManifest.views.map(row => [`${row.scene}/${row.queryStep}/${row.camera}`, row]));
assert.equal(rolloutViews.size, rolloutManifest.views.length);
assert.equal(rolloutManifest.planningInterval, 16);
assert.equal(rolloutManifest.policyK, 4);
let rolloutQueries = 0, rolloutPredictions = 0, rolloutImages = 0;
const uniqueScenes = [...new Map(groupScenes.map(s => [s.rolloutId,s])).values()];
for (const scene of uniqueScenes) {
  const source = rolloutManifest.scenes.find(row => row.id === scene.rolloutId);
  assert.equal(source.success, true);
  assert.equal(source.episodeFrames, source.steps+1);
  assert.deepEqual(scene.querySteps, source.queries, 'Do not omit source query times');
  assert.equal(scene.episodeFrames, source.episodeFrames);
  assert.ok(scene.querySteps.includes(scene.queryStep));
  assert.deepEqual(scene.querySteps, Array.from({length:Math.ceil(source.steps/16)}, (_,i)=>i*16));
  assert.equal(scene.videos, undefined, 'Only prediction-step images belong in the browser manifest');
  rolloutQueries += scene.querySteps.length;
  for (const query of scene.querySteps) for (const camera of scene.cameras) {
    const row = rolloutViews.get(`${scene.rolloutId}/${query}/${camera}`);
    assert.match(row.stateSHA256, /^[a-f0-9]{64}$/);
    const directory = `${scene.queryRoot}/q${String(query).padStart(6, '0')}`;
    const image = `${directory}/camera-${camera}.jpg`;
    assert.equal(createHash('sha256').update(fs.readFileSync(new URL(image, root))).digest('hex'), row.imageSHA256);
    rolloutImages++;
    for (const k of scene.budgets) {
      const data = json(`${directory}/k${k}-camera${camera}.json`);
      assert.equal(data.k, k); assert.equal(data.camera, camera); assert.equal(data.queryStep, query);
      assert.equal(data.image, image); assert.equal(data.imageSHA256, row.imageSHA256);
      assert.equal(data.sourceRGBSHA256, row.sourceRGBSHA256);
      assert.equal(data.sourceArchiveSHA256, row.archiveSHA256);
      assert.deepEqual([data.width, data.height, data.imageWidth, data.imageHeight], [128, 128, 2048, 2048]);
      assert.equal(data.rawPoints.length, k);
      assert.ok(data.rawPoints.every(xy => xy.length === 2 && xy.every(Number.isFinite)));
      assert.equal(data.groupPaths.length, k + 1);
      verifyGroupMap(data);
      rolloutPredictions++;
    }
  }
}
assert.equal(rolloutImages, rolloutViews.size);
let checked = 0;
const scenes = json('data/robocasa.json');
for (const scene of scenes) {
 assert.deepEqual(scene.budgets, BUDGETS);
 for (const k of scene.budgets) for (const camera of [0, 1, 2]) {
  const data = json(`data/robocasa/${scene.id}/k${k}-camera${camera}.json`);
  assert.deepEqual([data.imageWidth, data.imageHeight], [2048, 2048]);
  assert.ok(fs.existsSync(new URL(data.image, root)));
  assert.equal(data.rawPoints.length, k);
  assert.equal(data.groupPaths.length, k + 1);
  assert.equal(data.groupCounts.reduce((a, b) => a + b, 0), 128 * 128);
  for (let n = k; n <= 256; n++) {
    const result = upsample(data, n);
    assert.equal(result.points.length, n);
    assert.equal(result.parents.length, n);
    assert.equal(result.tracks.length, n);
    for (let g = 0; g < k; g++) {
      const index = result.parents.findIndex((parent, i) => parent === g &&
        result.points[i][0] === data.queryPoints[g][0] && result.points[i][1] === data.queryPoints[g][1]);
      assert.ok(index >= 0, 'Every representative must be retained at every N');
    }
    for (let i = 0; i < result.points.length; i++) {
      const parent = result.parents[i], xy = result.points[i];
      const isRepresentative = xy[0] === data.queryPoints[parent][0] && xy[1] === data.queryPoints[parent][1];
      assert.ok(isRepresentative || data.candidates[parent].pixels.includes(xy[1] * data.width + xy[0]), 'Every additional point belongs to its real ranked pool');
    }
    result.tracks.forEach(track => {
      assert.equal(track.length, 33);
      assert.ok(track.every(xy => xy === null || (xy.length === 2 && xy.every(Number.isFinite))));
      assert.ok(track[0], 'Every trajectory starts in front of the camera');
    });
    checked++;
  }
  assert.deepEqual(upsample(data, 999), data.savedPolicy);
  assert.equal(upsample(data, -1).points.length, k);
  const computed = upsample(data, 256, false);
  assert.deepEqual(computed.points, data.savedPolicy.points, `${scene.id}, K=${k}, camera=${camera}: exact policy pixels`);
  assert.deepEqual(computed.parents, data.savedPolicy.parents);
  const maxTrackError = Math.max(...computed.tracks.flatMap((track, i) => track.flatMap((xy, t) => {
    const saved = data.savedPolicy.tracks[i][t];
    assert.equal(xy === null, saved === null, 'Camera visibility agrees with saved policy');
    return xy ? xy.map((value, d) => Math.abs(value-saved[d])) : [0];
  })));
  assert.ok(maxTrackError < .01, `${scene.id}, K=${k}, camera=${camera}: projected policy track error ${maxTrackError}`);
 }
}
const html = fs.readFileSync(new URL('index.html', root), 'utf8');
for (const [, target] of html.matchAll(/(?:href|src)="([^"#]+)"/g)) {
  if (/^https?:/.test(target)) continue;
  assert.ok(fs.existsSync(new URL(target, root)), `Missing local resource: ${target}`);
}
const evidence = json('data/verification.json');
assert.ok(evidence.every(e => e.projectionMaxErrorPx < .01 && e.displacementMaxError < 1e-5 && e.savedPolicyPointsVerified === 256));
const results = json('data/efficiency.json');
assert.equal(results.rows.length, 44);
assert.equal(new Set(results.rows.map(r => r.id)).size, 44);
for (const row of results.rows) {
  assert.equal(row.episodes, 1200);
  assert.ok(Math.abs(row.success-row.successes/row.episodes*100) < 1e-10);
  assert.equal(row.totalGflops, row.generationGflops + row.selectionGflops);
  assert.equal(row.totalMs, row.generationMs + row.selectionMs);
  assert.ok(results.methods.some(m => m.id === row.method));
  assert.ok(Number.isFinite(row.hitRadius) && row.hitRadius > 0);
  if (row.method === 'sam3') assert.equal(row.selectionMs, 2241.1, 'Preserve author-confirmed PDF timing');
}
assert.ok(!html.includes('id="efficiency-k"'), 'The paper plot has no K slider');
for (const [id, ms] of [['dinov2-8',87.7],['dinov2-256',193.3],['dinov3-8',66.8],['dinov3-256',181.4]]) {
  assert.equal(results.rows.find(r => r.id === id).selectionMs, ms, 'Match PDF Table 4 timing');
}
// Independently transcribed from supplied PDF Table 22, in K order.
const published = [
  [4,19.3,10.1,4.0,2.7,51.6], [8,21.1,10.7,4.3,2.7,54.3],
  [16,24.8,11.8,4.7,2.8,55.0], [32,32.2,14.0,5.1,2.8,55.6],
  [64,47.2,18.5,6.4,3.1,57.1], [128,78.4,27.8,8.3,3.3,58.0],
  [256,144.8,47.8,13.5,3.9,56.7],
];
for (const expected of published) {
  const r = results.rows.find(r => r.method === 'ours' && r.k === expected[0]);
  assert.deepEqual([r.k,r.generationGflops,r.selectionGflops,r.generationMs,r.selectionMs,r.success].map(v=>Number(v.toFixed(1))), expected);
}
console.log(`Verified ${uniqueScenes.length} successful policy rollouts, ${rolloutQueries} query times at 16-step cadence, ${rolloutImages} query images, ${rolloutPredictions} exact selector maps, ${groupScenes.length} ranked examples, ${scenes.length} trajectory scenes, ${checked} upsampling settings, 44 performance points, local links, and source reconstruction checks.`);
