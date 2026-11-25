// Pure data operations shared by the interactive figures and verification script.
export const BUDGETS = [4, 8, 16, 32, 64, 128, 256];

export function groupColor(group, k) {
  // A muted blue separates the background mask from gray sinks and walls.
  if (group === k) return '#607fa6';
  // Color does not imply identity across separately predicted point budgets.
  const colors = ['#e65167', '#39c6e6', '#e3ac27', '#9766e7', '#3bc082', '#ea8038', '#d779c9', '#547de4'];
  return group < 8 ? colors[group] : `hsl(${(group * 137.508) % 360} 72% 56%)`;
}

export function allocate(data, n) {
  const extra = n - data.k;
  const order = Array.from({ length: data.k }, (_, i) => i)
    .sort((a, b) => data.poolCounts[b] - data.poolCounts[a] || a - b);
  const quotas = Array(data.k).fill(Math.floor(extra / data.k));
  for (let i = 0; i < extra % data.k; i++) quotas[order[i]]++;
  // Preserve every representative even when a group has an empty candidate pool.
  // Redistribute overflow only among groups with unused real candidate pixels.
  let overflow = 0;
  quotas.forEach((quota, i) => {
    const capacity = data.poolCapacities[i];
    if (quota > capacity) { overflow += quota - capacity; quotas[i] = capacity; }
  });
  while (overflow > 0) {
    let progress = false;
    for (const i of order) {
      if (quotas[i] < data.poolCapacities[i]) {
        quotas[i]++; overflow--; progress = true;
        if (!overflow) break;
      }
    }
    if (!progress) break;
  }
  // Match the original export: when a whole view has too few unique members,
  // repeat within nonempty groups after their unique pixels are exhausted.
  const nonempty = order.filter(i => data.poolCapacities[i] > 0);
  const repeatOrder = nonempty.length ? nonempty : order;
  for (let i = 0; i < overflow; i++) quotas[repeatOrder[i % repeatOrder.length]]++;
  return quotas;
}

export function upsample(data, requested, useSaved = true) {
  const n = Math.min(256, Math.max(data.k, Math.round(requested)));
  if (n === 256 && useSaved) return data.savedPolicy;
  const quotas = allocate(data, n);
  const result = { points: [], parents: [], tracks: [] };
  for (let g = 0; g < data.k; g++) {
    result.points.push(data.queryPoints[g]);
    result.parents.push(g);
    result.tracks.push(data.tracks[g]);
    for (let j = 0; j < quotas[g]; j++) {
      const pool = data.candidates[g];
      const index = j % pool.pixels.length;
      const pixel = pool.pixels[index];
      result.points.push(pool.pixels.length ? [pixel % data.width, Math.floor(pixel / data.width)] : data.queryPoints[g]);
      result.parents.push(g);
      result.tracks.push(pool.pixels.length ? pool.tracks[index] : data.tracks[g]);
    }
  }
  return result;
}
