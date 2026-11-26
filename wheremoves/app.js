import { groupColor, upsample } from './motion.js?v=20261124-background';
import { initEfficiency } from './results.js?v=20261126-pixels';

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
const NS = 'http://www.w3.org/2000/svg';
const ASSET_VERSION = '20261124-policy-rollouts';
const cache = new Map();
const sceneLists = new Map();
function currentScene(root) {
  const scenes = sceneLists.get(root) || [];
  return scenes.find(scene => scene.id === $('.robocasa-scene', root).value) || scenes[0];
}
function rolloutSteps(scene) {
  return scene.querySteps || [scene.queryStep];
}

function configureScene(root) {
  const scene = currentScene(root);
  const slider = $('input[id$="-k"]', root);
  slider.max = scene.budgets.length - 1;
  slider.value = Math.min(Number(slider.value), Number(slider.max));
  $('.camera-select', root).value = scene.defaultCamera;
  const description = $('.action-description', root) || $('.scene-caption', root);
  description.textContent = scene.instruction.replace(/^./, c => c.toUpperCase()) + '.';
  const progress = $('#group-progress', root);
  if (progress) {
    const queries = scene.querySteps || [scene.queryStep];
    const steps = rolloutSteps(scene);
    progress.max = steps.length - 1;
    progress.value = Math.max(0, steps.indexOf(scene.queryStep));
    $('.rollout-progress', root).hidden = queries.length < 2;
    $('.query-start', root).textContent = `Step ${queries[0]}`;
    $('.query-end', root).textContent = `Step ${queries.at(-1)}`;
    $('.prediction-blocks', root).replaceChildren(...queries.map((query, i) => {
      const block = document.createElement('span');
      block.className = 'prediction-block';
      block.dataset.query = query;
      block.style.flexGrow = i === 0 || i === queries.length - 1 ? .5 : 1;
      return block;
    }));
  }
  const ticks = $('.range-ticks', root);
  ticks.replaceChildren(...scene.budgets.map(k => {
    const button = document.createElement('button');
    button.dataset.k = k; button.textContent = k;
    button.addEventListener('click', () => {
      slider.value = scene.budgets.indexOf(k);
      slider.dispatchEvent(new Event('input', { bubbles: true }));
    });
    return button;
  }));
}
function loadJSON(url) {
  if (!cache.has(url)) cache.set(url, fetch(`${url}?v=${ASSET_VERSION}`, { cache: 'no-cache' }).then(response => {
    if (!response.ok) throw new Error(`Could not load ${url}`);
    return response.json();
  }).catch(error => { cache.delete(url); throw error; }));
  return cache.get(url);
}
function svgNode(tag, attrs = {}, text) {
  const node = document.createElementNS(NS, tag);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  if (text !== undefined) node.textContent = text;
  return node;
}
function sceneSvg(data, label) {
  const svg = svgNode('svg', { viewBox: `0 0 ${data.width} ${data.height}`, role: 'img', 'aria-label': label });
  svg.append(svgNode('image', { href: data.image, width: data.width, height: data.height }));
  return svg;
}
function point(svg, xy, group, k, radius = 1.35, interactive = true) {
  const p = svgNode('circle', { cx: xy[0] + .5, cy: xy[1] + .5, r: radius, fill: groupColor(group, k), stroke: '#fff', 'stroke-width': .35, 'data-group': group });
  if (interactive) p.append(svgNode('title', {}, `Pixel ${group + 1} · motion group ${group + 1}`));
  svg.append(p);
}
function selectionSvg(data) {
  const svg = sceneSvg(data, `${data.k} selected representative pixels`);
  data.rawPoints.forEach((xy, group) => point(svg, xy, group, data.k, data.k > 64 ? .85 : 1.35));
  return svg;
}
function groupsSvg(data) {
  const svg = sceneSvg(data, `${data.k} predicted motion groups and a background group`);
  data.groupPaths.forEach((d, group) => {
    const p = svgNode('path', { d, fill: groupColor(group, data.k), opacity: group === data.k ? .48 : .75, 'data-group': group, class: 'group-hit', 'shape-rendering': 'crispEdges' });
    p.append(svgNode('title', {}, group === data.k ? 'Background' : `Motion group ${group + 1}`));
    svg.append(p);
  });
  return svg;
}
function trajectorySvg(data, tracks, parents, frame, label) {
  const svg = sceneSvg(data, label);
  // Trails are vector paths built directly from the exported projected tracks.
  tracks.forEach((track, i) => {
    const color = groupColor(parents[i], data.k);
    let continuous = false;
    const path = track.slice(0, frame + 1).map(p => {
      if (!p) { continuous = false; return ''; }
      const command = continuous ? 'L' : 'M'; continuous = true;
      return `${command}${p[0] + .5} ${p[1] + .5}`;
    }).join('');
    svg.append(svgNode('path', { d: path, fill: 'none', stroke: color, 'stroke-width': tracks.length > 100 ? .35 : .55, 'stroke-opacity': .85 }));
    const p = track[frame];
    if (p) svg.append(svgNode('circle', { cx: p[0] + .5, cy: p[1] + .5, r: tracks.length > 100 ? .55 : 1, fill: color, stroke: '#fff', 'stroke-width': .12 }));
  });
  return svg;
}
function updateTicks(root, k) {
  $$('[data-k]', root).forEach(button => button.setAttribute('aria-pressed', String(Number(button.dataset.k) === k)));
}
function showError(root, error) {
  console.error(error);
  let message = $('.load-error', root);
  if (!message) { message = document.createElement('p'); message.className = 'load-error'; root.append(message); }
  message.textContent = 'The figure data could not be loaded. Refresh the page to try again.';
}
loadJSON('data/efficiency.json').then(data => initEfficiency($('#efficiency-demo'), data, uniqueIds))
  .catch(error => showError($('#efficiency-demo'), error));

// Collapsible side contents, including the method's nested explanations.
const nav = $('#contents');
const navToggle = $('#nav-toggle');
const navLinks = $$('a', nav);
const navTargets = navLinks.map(link => $(link.hash));
function setContentsOpen(open, restoreFocus = false) {
  document.body.classList.toggle('nav-open', open);
  navToggle.setAttribute('aria-expanded', String(open));
  nav.setAttribute('aria-hidden', String(!open));
  nav.inert = !open;
  if (restoreFocus) navToggle.focus();
}
navToggle.addEventListener('click', () => setContentsOpen(navToggle.getAttribute('aria-expanded') !== 'true'));
document.addEventListener('keydown', event => {
  if (event.key === 'Escape' && navToggle.getAttribute('aria-expanded') === 'true') setContentsOpen(false, true);
});
document.addEventListener('pointerdown', event => {
  if (!nav.contains(event.target) && !navToggle.contains(event.target)) setContentsOpen(false);
});
nav.addEventListener('click', event => {
  const link = event.target.closest('a');
  if (!link || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
  if (!matchMedia('(min-width: 1400px)').matches) setContentsOpen(false);
  const heading = $('h2, h3', $(link.hash));
  heading.setAttribute('tabindex', '-1');
  heading.focus({ preventScroll: true });
});
let navFrame;
function updateCurrentSection() {
  navFrame = undefined;
  const readingLine = Math.max(80, window.innerHeight * .18);
  let current = -1;
  navTargets.forEach((section, index) => {
    if (section.getBoundingClientRect().top <= readingLine) current = index;
  });
  if (window.scrollY > 0 && window.scrollY + window.innerHeight >= document.documentElement.scrollHeight - 2) current = navTargets.length - 1;
  navLinks.forEach((link, index) => {
    link.classList.toggle('active', index === current);
    if (index === current) link.setAttribute('aria-current', 'location');
    else link.removeAttribute('aria-current');
  });
}
function scheduleContentsUpdate() {
  if (navFrame === undefined) navFrame = requestAnimationFrame(updateCurrentSection);
}
window.addEventListener('scroll', scheduleContentsUpdate, { passive: true });
window.addEventListener('resize', scheduleContentsUpdate);
window.addEventListener('load', scheduleContentsUpdate);
updateCurrentSection();

// Scope SVG IDs because independent PDF exports reuse identifiers.
function uniqueIds(svg, prefix) {
  const ids = new Map();
  $$('[id]', svg).forEach(node => { ids.set(node.id, prefix + node.id); node.id = prefix + node.id; });
  for (const node of [svg, ...$$('*', svg)]) for (const attr of [...node.attributes]) {
    let value = attr.value;
    if (value.startsWith('#') && ids.has(value.slice(1))) value = '#' + ids.get(value.slice(1));
    value = value.replace(/url\(#([^)]+)\)/g, (match, id) => ids.has(id) ? `url(#${ids.get(id)})` : match);
    if (value !== attr.value) node.setAttributeNS(attr.namespaceURI, attr.name, value);
  }
}
for (const figure of $$('[data-vector]')) {
  fetch(`assets/figures/${figure.dataset.vector}.svg`).then(r => {
    if (!r.ok) throw new Error('Figure not found'); return r.text();
  }).then(text => {
    const svg = new DOMParser().parseFromString(text, 'image/svg+xml').documentElement;
    uniqueIds(svg, figure.dataset.vector + '-');
    svg.setAttribute('role', 'img'); svg.setAttribute('aria-label', $('.vector-content', figure).getAttribute('aria-label'));
    // The supplied PDFs use Times/Nimbus Roman, with Computer Modern math.
    $$('text', svg).forEach(t => { const f = t.getAttribute('font-family'); if (f?.includes('Computer Modern')) t.setAttribute('font-family', 'Latin Modern, serif'); });
    $('.vector-content', figure).replaceChildren(svg);
  }).catch(error => showError(figure, error));
}

// First figure: independent representative sets and their corresponding groups.
const groupRoot = $('#group-demo');
let groupData, groupVersion = 0, pinned = -1;
function highlight(group = -1) {
  if (!groupData) return;
  $$('[data-group]', groupRoot).forEach(node => {
    const g = Number(node.dataset.group);
    const base = node.tagName.toLowerCase() === 'path' ? (g === groupData.k ? .48 : .75) : 1;
    node.style.opacity = group < 0 || g === group ? base : .1;
  });
  $('.group-status', groupRoot).textContent = group < 0
    ? 'Hover a pixel or motion group to see its correspondence. Click to keep it highlighted.'
    : `${group === groupData.k ? 'Background' : `Motion group ${group + 1}`} · ${groupData.groupCounts[group].toLocaleString()} pixels${pinned === group ? ' · selected' : ''}`;
}
async function renderGroups() {
  const version = ++groupVersion;
  const scene = currentScene(groupRoot);
  if (!scene) return;
  const k = scene.budgets[Number($('#group-k').value)], camera = Number($('.camera-select', groupRoot).value);
  const queries = scene.querySteps || [scene.queryStep];
  const progress = Number($('#group-progress').value);
  const query = rolloutSteps(scene)[progress];
  const queryIndex = queries.indexOf(query);
  const directory = scene.queryRoot ? `${scene.queryRoot}/q${String(query).padStart(6, '0')}` : `data/groups/${scene.id}`;
  groupData = undefined;
  groupRoot.setAttribute('aria-busy', 'true');
  $('.group-status', groupRoot).textContent = `Loading predictions for step ${query}…`;
  try {
    const data = await loadJSON(`${directory}/k${k}-camera${camera}.json`);
    if (version !== groupVersion) return;
    // Decode the matching image before replacing either side of the pair.
    const image = new Image();
    image.src = data.image;
    await image.decode();
    if (version !== groupVersion) return;
    groupData = data; pinned = -1;
    $('.points-view', groupRoot).replaceChildren(selectionSvg(data));
    $('.groups-view', groupRoot).replaceChildren(groupsSvg(data));
    $('.k-label', groupRoot).textContent = k;
    $('.groups-label', groupRoot).textContent = `${k} + background`;
    const select = $('.group-select', groupRoot);
    select.disabled = false;
    $('.clear-highlight', groupRoot).disabled = false;
    select.replaceChildren(new Option('All groups', -1), ...Array.from({ length: k }, (_, i) => new Option(`Group ${i + 1}`, i)), new Option('Background', k));
    groupRoot.dataset.prediction = 'true';
    highlight();
    $('output[for=group-k]', groupRoot).textContent = k;
    $('output[for=group-progress]', groupRoot).textContent = `Step ${query} · prediction ${queryIndex + 1} / ${queries.length}`;
    $('#group-progress').setAttribute('aria-valuetext', `Prediction ${queryIndex + 1} of ${queries.length}, rollout step ${query}`);
    groupRoot.dataset.queryStep = query;
    groupRoot.dataset.camera = camera;
    groupRoot.dataset.budget = k;
    const blockStart = queries.findLast(q => q <= query);
    $$('.prediction-block', groupRoot).forEach(block => {
      const start = Number(block.dataset.query);
      block.classList.toggle('is-prediction', start === query);
      block.classList.toggle('is-current', start === blockStart);
    });
    $('.load-error', groupRoot)?.remove();
    $('#group-k').setAttribute('aria-valuetext', `${k} selected pixels`);
    updateTicks(groupRoot, k);
    $('.previous-query', groupRoot).disabled = !queries.some(q => q < query);
    $('.next-query', groupRoot).disabled = !queries.some(q => q > query);
  } catch (error) { if (version === groupVersion) showError(groupRoot, error); }
  finally { if (version === groupVersion) groupRoot.setAttribute('aria-busy', 'false'); }
}
$('#group-k').addEventListener('input', renderGroups);
$('#group-progress').addEventListener('input', renderGroups);
for (const direction of ['previous', 'next']) $('.' + direction + '-query', groupRoot).addEventListener('click', () => {
  const scene = currentScene(groupRoot);
  const slider = $('#group-progress');
  const queries = scene.querySteps;
  const steps = rolloutSteps(scene);
  const current = steps[Number(slider.value)];
  const target = direction === 'previous' ? queries.findLast(q => q < current) : queries.find(q => q > current);
  if (target === undefined) return;
  slider.value = steps.indexOf(target);
  renderGroups();
});
$('.camera-select', groupRoot).addEventListener('change', renderGroups);
$('.robocasa-scene', groupRoot).addEventListener('change', () => { configureScene(groupRoot); renderGroups(); });
groupRoot.addEventListener('pointerover', event => { const target = event.target.closest('[data-group]'); if (target && pinned < 0) highlight(Number(target.dataset.group)); });
groupRoot.addEventListener('pointerout', event => { if (event.target.closest('[data-group]') && pinned < 0) highlight(); });
groupRoot.addEventListener('click', event => {
  const target = event.target.closest('[data-group]');
  if (target) { const g = Number(target.dataset.group); pinned = pinned === g ? -1 : g; $('.group-select', groupRoot).value = pinned; highlight(pinned); }
});
$('.group-select', groupRoot).addEventListener('change', event => { pinned = Number(event.target.value); highlight(pinned); });
$('.clear-highlight', groupRoot).addEventListener('click', () => { pinned = -1; $('.group-select', groupRoot).value = -1; highlight(); });

// Second figure: independent K and N, preserving the real forecast displacements.
const upRoot = $('#upsample-demo');
let upData, upVersion = 0, expanded, playing = false, lastFrame = 0, animationId;
function renderTrajectories() {
  if (!upData || !expanded) return;
  const frame = Number($('#time').value);
  $('.sparse-view', upRoot).replaceChildren(trajectorySvg(upData, upData.tracks, upData.tracks.map((_, i) => i), frame, `${upData.k} generated trajectories at frame ${frame}`));
  $('.dense-view', upRoot).replaceChildren(trajectorySvg(upData, expanded.tracks, expanded.parents, frame, `${expanded.tracks.length} expanded trajectories at frame ${frame}`));
  $('output[for=time]').textContent = `${frame} / 32`;
}
function renderN() {
  if (!upData) return;
  const slider = $('#upsample-n');
  slider.min = upData.k;
  slider.value = Math.max(upData.k, Number(slider.value));
  slider.disabled = upData.k === 256;
  const n = Number(slider.value);
  expanded = upsample(upData, n);
  $('.n-min', upRoot).textContent = upData.k;
  $('.n-label', upRoot).textContent = n;
  $('output[for=upsample-n]').textContent = n;
  const factor = n / upData.k;
  $('.budget-summary', upRoot).innerHTML = factor > 1
    ? `<strong>${Number(factor.toFixed(2))}×</strong> fewer generated trajectories for ${n} displayed pixels.`
    : `<strong>${n} generated trajectories.</strong> No upsampling at this setting.`;
  renderTrajectories();
}
async function renderUpsampling() {
  const version = ++upVersion;
  const scene = currentScene(upRoot);
  if (!scene) return;
  const k = scene.budgets[Number($('#upsample-k').value)], camera = Number($('.camera-select', upRoot).value);
  try {
    const data = await loadJSON(`data/robocasa/${scene.id}/k${k}-camera${camera}.json`);
    if (version !== upVersion) return;
    upData = data;
    $('.k-label', upRoot).textContent = k;
    $('output[for=upsample-k]').textContent = k;
    $('#upsample-k').setAttribute('aria-valuetext', `${k} selected pixels`);
    updateTicks(upRoot, k); renderN();
  } catch (error) { if (version === upVersion) showError(upRoot, error); }
}
$('#upsample-k').addEventListener('input', renderUpsampling);
$('#upsample-n').addEventListener('input', renderN);
$('.camera-select', upRoot).addEventListener('change', renderUpsampling);
$('.robocasa-scene', upRoot).addEventListener('change', () => { stopPlayback(); configureScene(upRoot); renderUpsampling(); });
$('#time').addEventListener('input', () => { stopPlayback(); renderTrajectories(); });
function stopPlayback() { playing = false; cancelAnimationFrame(animationId); $('.play-toggle').textContent = 'Play'; $('.play-toggle').setAttribute('aria-label', 'Play predicted trajectories'); }
function animate(time) {
  if (!playing) return;
  if (time - lastFrame > 100) { $('#time').value = (Number($('#time').value) + 1) % 33; renderTrajectories(); lastFrame = time; }
  animationId = requestAnimationFrame(animate);
}
$('.play-toggle').addEventListener('click', () => {
  if (playing) return stopPlayback();
  playing = true; $('.play-toggle').textContent = 'Pause'; $('.play-toggle').setAttribute('aria-label', 'Pause predicted trajectories');
  animationId = requestAnimationFrame(animate);
});
document.addEventListener('visibilitychange', () => { if (document.hidden) stopPlayback(); });
new IntersectionObserver(entries => { if (!entries[0].isIntersecting) stopPlayback(); }).observe(upRoot);

for (const [root, manifest, render] of [
  [groupRoot, 'data/groups.json', renderGroups],
  [upRoot, 'data/robocasa.json', renderUpsampling],
]) {
  loadJSON(manifest).then(scenes => {
    sceneLists.set(root, scenes);
    $('.robocasa-scene', root).replaceChildren(...scenes.map(scene => new Option(scene.name, scene.id)));
    configureScene(root);
    render();
  }).catch(error => showError(root, error));
}

$('#copy-bibtex').addEventListener('click', async () => {
  const status = $('#citation-status');
  try {
    await navigator.clipboard.writeText($('#bibtex-entry').textContent.trim());
    status.textContent = 'Copied.';
  } catch {
    status.textContent = 'Select and copy the citation below.';
  }
});
