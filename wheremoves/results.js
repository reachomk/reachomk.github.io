// Preserve the paper SVG; invisible circle targets provide hover details.
const NS = 'http://www.w3.org/2000/svg';
const number = value => value.toLocaleString('en-US', {minimumFractionDigits: 1, maximumFractionDigits: 1});
function node(tag, attrs = {}) {
  const el = document.createElementNS(NS, tag);
  for (const [key, value] of Object.entries(attrs)) el.setAttribute(key, value);
  return el;
}

export async function initEfficiency(root, data, uniqueIds) {
  const response = await fetch('assets/figures/efficiency.svg');
  if (!response.ok) throw new Error('Could not load the original performance figure');
  const svg = new DOMParser().parseFromString(await response.text(), 'image/svg+xml').documentElement;
  uniqueIds(svg, 'performance-');
  // The PDF crop retains off-crop page elements. Accessible labels live on the
  // measured circle targets, while the original visible SVG text stays selectable.
  for (const child of svg.children) child.setAttribute('aria-hidden', 'true');
  svg.classList.add('performance-svg');
  svg.setAttribute('role', 'group');
  svg.setAttribute('aria-label', 'Original RoboCasa success and compute plot. Hover, tap, or focus a circle for details.');
  const methods = new Map(data.methods.map(m => [m.id, m]));
  const axis = data.plotCoordinates;
  const positions = data.rows.map(row => ({row, x: axis.x4 + Math.log2(row.k / 4) * axis.xPerDoubling, y: axis.y0 - row.success * axis.yPerPercent}));
  const chart = root.querySelector('.efficiency-chart');
  const overlay = node('g', {class: 'performance-targets'});
  const marker = node('circle', {r: 2.7, fill: 'none', stroke: '#20241f', 'stroke-width': .65, class: 'performance-marker', visibility: 'hidden'});
  const tooltip = document.createElement('div');
  tooltip.className = 'plot-tooltip'; tooltip.id = 'efficiency-tooltip'; tooltip.setAttribute('role', 'tooltip'); tooltip.hidden = true;
  let pinned = null, focused = false, current = null;

  function hide() { current = null; tooltip.hidden = true; marker.setAttribute('visibility', 'hidden'); }
  function place() {
    if (!current) return;
    const width = chart.clientWidth, scale = svg.clientWidth / axis.width;
    const x = current.x * scale, y = current.y * scale;
    const left = Math.max(4, Math.min(width-tooltip.offsetWidth-4, x-tooltip.offsetWidth/2));
    const top = y > tooltip.offsetHeight + 16 ? y-tooltip.offsetHeight-12 : y+16;
    tooltip.style.left = `${left}px`; tooltip.style.top = `${top}px`;
  }
  function show(position) {
    current = position;
    // Grid and DINOv3 at K=256 share exactly the same plotted coordinate.
    // Report both instead of silently making one inaccessible under the other.
    const coincident = positions.filter(p => Math.hypot(p.x-position.x, p.y-position.y) < .02);
    tooltip.replaceChildren(...coincident.map(({row}) => {
      const section = document.createElement('section'), heading = document.createElement('strong');
      heading.textContent = `${methods.get(row.method).name} · K = ${row.k.toLocaleString()}`;
      section.append(heading);
      const metrics = document.createElement('dl');
      for (const [label, value] of [['GFLOPs', number(row.totalGflops)], ['Time', number(row.totalMs)+' ms'], ['Success', number(row.success)+'%']]) {
        const item = document.createElement('div'), dt = document.createElement('dt'), dd = document.createElement('dd');
        dt.textContent = label; dd.textContent = value; item.append(dt, dd); metrics.append(item);
      }
      section.append(metrics);
      const stage = value => value === 0 ? '≈0' : number(value);
      const details = document.createElement('p');
      details.textContent = `Generation: ${number(row.generationGflops)} GFLOPs · ${number(row.generationMs)} ms`;
      const selection = document.createElement('p');
      selection.textContent = `Selection: ${stage(row.selectionGflops)} GFLOPs · ${stage(row.selectionMs)} ms`;
      const trials = document.createElement('p');
      trials.className = 'tooltip-trials'; trials.textContent = `${row.successes.toLocaleString()} / ${row.episodes.toLocaleString()} rollouts${row.k > 256 ? ' · 256 retained points' : ''}`;
      section.append(details, selection, trials); return section;
    }));
    tooltip.hidden = false;
    marker.setAttribute('cx', position.x); marker.setAttribute('cy', position.y); marker.setAttribute('visibility', 'visible');
    place();
  }
  function hit(event) {
    const point = new DOMPoint(event.clientX, event.clientY).matrixTransform(svg.getScreenCTM().inverse());
    const candidates = positions.filter(p => Math.hypot(p.x-point.x, p.y-point.y) <= Math.max(p.row.hitRadius, 2.3));
    return candidates.sort((a,b) => Math.hypot(a.x-point.x,a.y-point.y)-Math.hypot(b.x-point.x,b.y-point.y))[0];
  }
  function toggle(position) {
    if (pinned === position.row.id) { pinned = null; hide(); }
    else { pinned = position.row.id; show(position); }
  }
  for (const position of positions) {
    const row = position.row, method = methods.get(row.method);
    const target = node('circle', {cx:position.x, cy:position.y, r:Math.max(row.hitRadius,2.3), fill:'transparent', class:'performance-target', 'data-result':row.id,
      role:'button', tabindex:0, 'aria-describedby':'efficiency-tooltip', 'aria-label':`${method.name}, K ${row.k}: ${number(row.success)}% success, ${number(row.totalGflops)} GFLOPs, ${number(row.totalMs)} ms`});
    target.addEventListener('focus', () => { focused = true; show(position); });
    target.addEventListener('blur', () => { focused = false; if (!pinned) hide(); });
    target.addEventListener('keydown', event => {
      if (['Enter',' '].includes(event.key)) { event.preventDefault(); toggle(position); }
      if (event.key === 'Escape') { pinned = null; hide(); }
    });
    overlay.append(target);
  }
  svg.append(overlay, marker);
  svg.addEventListener('pointermove', event => {
    if (pinned || event.pointerType !== 'mouse') return;
    const position = hit(event); if (position) show(position); else if (!focused) hide();
  });
  svg.addEventListener('pointerleave', () => { if (!pinned && !focused) hide(); });
  svg.addEventListener('click', event => { const position = hit(event); if (position) toggle(position); });
  document.addEventListener('pointerdown', event => { if (!chart.contains(event.target)) { pinned = null; hide(); } });
  document.addEventListener('keydown', event => { if (event.key === 'Escape') { pinned = null; hide(); } });
  new ResizeObserver(place).observe(chart);
  chart.replaceChildren(svg, tooltip);
}
