// Port 1:1 di normalizza_sequenza / calcola_direzione / calcola_distanze /
// estendi_sequenza da sportello_offline.py (Conformer v4, INPUT_DIM_EX=744).
// Ogni frame grezzo e' un Float32Array(360): lh(63) + rh(63) + pose(24) + face(210),
// stesso ordine di estrai_keypoints() Python.

export const DIST_DIM = 18;
export const DIR_DIM  = 6;
export const N_FACE   = 70;

export const COPPIE_DIST = [
  [4,8],[4,12],[4,16],[4,20],[8,12],[8,16],[8,20],
  [12,16],[12,20],[0,8],[0,12],[0,4],[4,0],[8,4],
  [12,4],[16,4],[20,4],[0,20],
];

function handPresenteX(f, offset) {
  for (let i = 0; i < 21; i++) if (f[offset + i * 3] !== 0) return true;
  return false;
}

function normalizzaManoInPlace(f, offset) {
  if (!handPresenteX(f, offset)) return;
  const wx = f[offset], wy = f[offset + 1], wz = f[offset + 2];
  for (let i = 0; i < 21; i++) {
    f[offset + i * 3]     -= wx;
    f[offset + i * 3 + 1] -= wy;
    f[offset + i * 3 + 2] -= wz;
  }
  const dx = f[offset + 5 * 3]     - f[offset + 17 * 3];
  const dy = f[offset + 5 * 3 + 1] - f[offset + 17 * 3 + 1];
  const dz = f[offset + 5 * 3 + 2] - f[offset + 17 * 3 + 2];
  const norma = Math.sqrt(dx * dx + dy * dy + dz * dz) + 1e-8;
  for (let i = 0; i < 63; i++) f[offset + i] /= norma;
}

function normalizzaPoseInPlace(f) {
  const off = 126;
  if (f[off] === 0 || f[off + 3] === 0) return; // pose[0,0] e pose[1,0] devono essere != 0
  const cx = (f[off] + f[off + 3]) / 2;
  const cy = (f[off + 1] + f[off + 4]) / 2;
  const cz = (f[off + 2] + f[off + 5]) / 2;
  for (let i = 0; i < 8; i++) {
    f[off + i * 3]     -= cx;
    f[off + i * 3 + 1] -= cy;
    f[off + i * 3 + 2] -= cz;
  }
  const dx = f[off] - f[off + 3], dy = f[off + 1] - f[off + 4], dz = f[off + 2] - f[off + 5];
  const dist = Math.sqrt(dx * dx + dy * dy + dz * dz) + 1e-8;
  for (let i = 0; i < 24; i++) f[off + i] /= dist;
}

function normalizzaVisoInPlace(f) {
  const off = 150;
  if (f[off + 16 * 3] === 0) return; // face[16,0] != 0 (punto 16 = landmark 4, naso)
  const presenti = new Array(N_FACE);
  for (let i = 0; i < N_FACE; i++) {
    presenti[i] = f[off + i * 3] !== 0 || f[off + i * 3 + 1] !== 0 || f[off + i * 3 + 2] !== 0;
  }
  const ox = f[off + 16 * 3], oy = f[off + 16 * 3 + 1], oz = f[off + 16 * 3 + 2];
  const dx0 = f[off] - f[off + 4 * 3], dy0 = f[off + 1] - f[off + 4 * 3 + 1], dz0 = f[off + 2] - f[off + 4 * 3 + 2];
  const dist = Math.sqrt(dx0 * dx0 + dy0 * dy0 + dz0 * dz0) + 1e-8;
  for (let i = 0; i < N_FACE; i++) {
    if (presenti[i]) {
      f[off + i * 3]     = (f[off + i * 3]     - ox) / dist;
      f[off + i * 3 + 1] = (f[off + i * 3 + 1] - oy) / dist;
      f[off + i * 3 + 2] = (f[off + i * 3 + 2] - oz) / dist;
    } else {
      f[off + i * 3] = 0; f[off + i * 3 + 1] = 0; f[off + i * 3 + 2] = 0;
    }
  }
}

// Normalizza mani + pose + viso per ogni frame. Non modifica seqRaw (ritorna copie).
export function normalizzaSequenza(seqRaw) {
  return seqRaw.map((frame) => {
    const f = Float32Array.from(frame);
    normalizzaManoInPlace(f, 0);
    normalizzaManoInPlace(f, 63);
    normalizzaPoseInPlace(f);
    normalizzaVisoInPlace(f);
    return f;
  });
}

// Direzione/velocita' dei polsi — calcolata sui dati GREZZI (non normalizzati),
// esattamente come in Python (r = calcola_direzione(seq) prima della normalizzazione).
export function calcolaDirezione(seqRaw) {
  const T = seqRaw.length;
  const out = [];
  let prev = null;
  for (let t = 0; t < T; t++) {
    const f = seqRaw[t];
    const d = new Float32Array(DIR_DIM);
    if (t > 0) {
      const dlx = f[0] - prev[0], dly = f[1] - prev[1], dlz = f[2] - prev[2];
      const nl  = Math.sqrt(dlx * dlx + dly * dly + dlz * dlz) + 1e-8;
      d[0] = dlx / nl; d[1] = dly / nl; d[2] = dlz / nl;
      const drx = f[63] - prev[63], dry = f[64] - prev[64], drz = f[65] - prev[65];
      const nr  = Math.sqrt(drx * drx + dry * dry + drz * drz) + 1e-8;
      d[3] = drx / nr; d[4] = dry / nr; d[5] = drz / nr;
    }
    out.push(d);
    prev = f;
  }
  return out;
}

function any63(arr, offset) {
  for (let i = 0; i < 63; i++) if (arr[offset + i] !== 0) return true;
  return false;
}

function var63(arr, offset) {
  let mean = 0;
  for (let i = 0; i < 63; i++) mean += arr[offset + i];
  mean /= 63;
  let v = 0;
  for (let i = 0; i < 63; i++) { const d = arr[offset + i] - mean; v += d * d; }
  return v / 63;
}

// Distanze fra punti della mano "attiva" (quella con varianza maggiore se
// entrambe presenti) — calcolate sulla sequenza GIA' normalizzata.
export function calcolaDistanze(seqNorm) {
  const T = seqNorm.length;
  const out = [];
  for (let t = 0; t < T; t++) {
    const f = seqNorm[t];
    const hasLh = any63(f, 0), hasRh = any63(f, 63);
    let useLh = false;
    if (hasLh) useLh = !hasRh || var63(f, 0) >= var63(f, 63);
    const useRh = hasRh && !useLh;

    const d = new Float32Array(DIST_DIM);
    if (useLh || useRh) {
      const off = useLh ? 0 : 63;
      for (let k = 0; k < COPPIE_DIST.length; k++) {
        const [i, j] = COPPIE_DIST[k];
        const dx = f[off + i * 3]     - f[off + j * 3];
        const dy = f[off + i * 3 + 1] - f[off + j * 3 + 1];
        const dz = f[off + i * 3 + 2] - f[off + j * 3 + 2];
        d[k] = Math.sqrt(dx * dx + dy * dy + dz * dz);
      }
    }
    out.push(d);
  }
  return out;
}

// Combina tutto: [seqNorm(360), delta(360), direzione(6), distanze(18)] = 744
// per ogni frame — stesso ordine di INPUT_DIM_EX in sportello_offline.py.
export function estendiSequenza(seqRaw) {
  const direzione = calcolaDirezione(seqRaw);
  const seqNorm   = normalizzaSequenza(seqRaw);
  const T = seqNorm.length;

  const delta = [];
  for (let t = 0; t < T; t++) {
    const d = new Float32Array(360);
    if (t > 0) for (let i = 0; i < 360; i++) d[i] = seqNorm[t][i] - seqNorm[t - 1][i];
    delta.push(d);
  }

  const distanze = calcolaDistanze(seqNorm);

  const out = [];
  for (let t = 0; t < T; t++) {
    const row = new Float32Array(744);
    row.set(seqNorm[t], 0);
    row.set(delta[t], 360);
    row.set(direzione[t], 720);
    row.set(distanze[t], 726);
    out.push(row);
  }
  return out;
}
