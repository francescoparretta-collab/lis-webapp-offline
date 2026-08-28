// Versione OFFLINE: librerie e modelli caricati da file locali (vendor/),
// non da internet — dopo il primo avvio (che li mette in cache) funziona
// anche senza connessione.
import {
  FilesetResolver,
  HandLandmarker,
  FaceLandmarker,
  PoseLandmarker
} from "./vendor/tasks-vision/vision_bundle.mjs";
import { estendiSequenza } from './features.js';

// ── Parametri di calibrazione — IDENTICI a app_conformer5.py ──
// (calibrati su log reale 2026-07-29, 7101 predizioni: 0.85 scartava
// l'81% delle lettere; il margine globale non scattava mai con soglia
// >=0.75 e softmax, quindi eliminato — resta solo per le coppie ambigue)
const SEQ_LEN_LETTERA = 8;
const SEQ_LEN_PAROLA  = 10;
const SOGLIA_LETTERA  = 0.75;
const SOGLIA_PAROLA   = 0.90;
const MARGINE_AMBIGUE = 0.60;
const CONFERME_MIN    = 2;
const MAX_SEGNI       = 15;
const SLIDE_STEP      = 2;

const LETTERE = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('');
const PAROLE  = ['aiutare','capire','casa','mangiare','scusa','io','tu','no',
                 'si','male','nome','buono','giorno','chi','dove','mio','tuo',
                 'cosa','mamma','papa','figlio','fratello','sorella','zio',
                 'regalare','suo','andare','venire','venire1','domani','oggi',
                 'ieri','rosso','giallo','verde','blu','marrone','scuro',
                 'chiaro','segno','conoscere','guardare','insegnare','quale',
                 'cane','gatto','cavallo','mucca','pesce','uccello',
                 'coniglio','piacere','piacere_no','ce','non_ce','animale'];
const SEGNI = [...LETTERE, ...PAROLE]; // 82 classi, stesso ordine del training

function coppiaKey(a, b) { return [a, b].sort().join('|'); }
const COPPIE_AMBIGUE = new Set([
  ['A','S'],['R','U'],['M','N'],['D','G'],['H','U'],['V','K'],
  ['V','N'],['G','T'],['I','Y'],['Y','E'],['Q','no'],
  ['H','nome'],['R','W'],['U','W'],['Z','X'],['M','I'],['N','I'],
  ['M','Y'],['E','F'],
  ['H','pesce'],['domani','chi'],['chi','rosso'],['mio','tuo'],['tu','quale'],
  ['ieri','chi'],['capire','male'],['suo','tuo'],
].map(([a, b]) => coppiaKey(a, b)));

const PAUSA_DOPO_CONFERMA = 5;  // frame di raffreddamento dopo un segno confermato (come pausa=5 in Python)
const PAUSA_FRASE         = 15; // frame senza mani prima di chiudere la frase corrente in una bolla
const TOLLERANZA_PERSA    = 10; // frame di tolleranza: la mano puo' sparire un istante
                                 // (vicino a spalla/testa) senza azzerare il buffer

let modello;
let bufferGrezzo = []; // frame raw Float32Array(360), finestra scorrevole
let candidato = '';
let conferme  = 0;
let manoAssente = true;
let frase = [];        // segni della frase in corso (bolla "pending")
let messaggi = [];     // bolle gia' chiuse
let frameSenzaMani = 0;
let manoPersa = 0;
let pausa = 0;

// ── Indici landmark — IDENTICI a raccolta_dati_5.py / sportello_offline.py ──
const FACE_IDX_V1  = [33,133,159,145,263,362,386,374,61,291,13,14,17,84,314,1,4];
const LABBRA_IDX    = [61,146,91,181,84,17,314,405,321,375,291,409,270,269,267,0,37,39,40,185,
                        78,95,88,178,87,14,317,402,318,324,308,415,310,311,312,13,82,81,80,191];
const SOPRACC_IDX   = [70,63,105,66,107,46,53,52,65,55,
                        300,293,334,296,336,276,283,282,295,285];
const POSE_IDX      = [11,12,13,14,15,16,23,24];

const FACE_V1_SET = new Set(FACE_IDX_V1);
const FACE_EXTRA  = [...LABBRA_IDX, ...SOPRACC_IDX].filter(i => !FACE_V1_SET.has(i));
const FACE_IDX    = [...FACE_IDX_V1, ...FACE_EXTRA]; // 70 punti totali, stesso ordine del training

const HAND_CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],
  [0,5],[5,6],[6,7],[7,8],
  [5,9],[9,10],[10,11],[11,12],
  [9,13],[13,14],[14,15],[15,16],
  [13,17],[17,18],[18,19],[19,20],
  [0,17]
];
const POSE_CONNECTIONS = [[11,12],[11,13],[13,15],[12,14],[14,16],[11,23],[12,24],[23,24]];

const video     = document.getElementById('video');
const overlay   = document.getElementById('overlay');
const ctx       = overlay.getContext('2d');
const statoEl   = document.getElementById('stato');
const loadingEl = document.getElementById('loading');

let handLandmarker, faceLandmarker, poseLandmarker;
let workCanvas, workCtx;

function setStato(testo, classe) {
  statoEl.textContent = testo;
  statoEl.className = classe;
}

async function initCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: 'user', width: { ideal: 480 }, height: { ideal: 360 } },
    audio: false
  });
  video.srcObject = stream;
  await new Promise(resolve => { video.onloadedmetadata = resolve; });
  await video.play();

  overlay.width  = video.videoWidth;
  overlay.height = video.videoHeight;

  workCanvas = document.createElement('canvas');
  workCanvas.width  = video.videoWidth;
  workCanvas.height = video.videoHeight;
  workCtx = workCanvas.getContext('2d', { willReadFrequently: true });
}

async function initModelli() {
  const vision = await FilesetResolver.forVisionTasks('./vendor/tasks-vision/wasm');

  // Soglie allineate a sportello_offline.py (min_detection=0.4, min_tracking=0.3)
  // — piu' permissive del default 0.5, per tollerare meglio condizioni difficili
  // (mano vicino al corpo, movimento veloce), come gia' validato sul PC.
  handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: './vendor/models/hand_landmarker.task',
      delegate: 'GPU'
    },
    runningMode: 'VIDEO',
    numHands: 2,
    minHandDetectionConfidence: 0.4,
    minHandPresenceConfidence: 0.4,
    minTrackingConfidence: 0.3
  });

  faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: './vendor/models/face_landmarker.task',
      delegate: 'GPU'
    },
    runningMode: 'VIDEO',
    numFaces: 1,
    minFaceDetectionConfidence: 0.4,
    minFacePresenceConfidence: 0.4,
    minTrackingConfidence: 0.3
  });

  poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: './vendor/models/pose_landmarker_lite.task',
      delegate: 'GPU'
    },
    runningMode: 'VIDEO',
    numPoses: 1,
    minPoseDetectionConfidence: 0.4,
    minPosePresenceConfidence: 0.4,
    minTrackingConfidence: 0.3
  });
}

// ── Disegno ──────────────────────────────────────────────────
function disegnaPunto(x, y, colore, raggio = 3) {
  ctx.beginPath();
  ctx.arc(x * overlay.width, y * overlay.height, raggio, 0, Math.PI * 2);
  ctx.fillStyle = colore;
  ctx.fill();
}

function disegnaLinea(a, b, colore, larghezza = 2) {
  ctx.beginPath();
  ctx.moveTo(a.x * overlay.width, a.y * overlay.height);
  ctx.lineTo(b.x * overlay.width, b.y * overlay.height);
  ctx.strokeStyle = colore;
  ctx.lineWidth = larghezza;
  ctx.stroke();
}

function disegnaMano(landmarks, colore) {
  for (const [i, j] of HAND_CONNECTIONS) disegnaLinea(landmarks[i], landmarks[j], colore, 2);
  for (const p of landmarks) disegnaPunto(p.x, p.y, colore, 3);
}

function disegnaPose(landmarks) {
  for (const [i, j] of POSE_CONNECTIONS) {
    if (landmarks[i] && landmarks[j]) disegnaLinea(landmarks[i], landmarks[j], 'rgba(200,200,200,0.6)', 2);
  }
}

function disegnaViso(landmarks) {
  for (const i of FACE_IDX_V1)  disegnaPunto(landmarks[i].x, landmarks[i].y, '#7fe08a', 2);
  for (const i of LABBRA_IDX)   disegnaPunto(landmarks[i].x, landmarks[i].y, '#ff8a00', 2);
  for (const i of SOPRACC_IDX)  disegnaPunto(landmarks[i].x, landmarks[i].y, '#00e0ff', 2);
}

// ── Modello ──────────────────────────────────────────────────
async function initModello() {
  modello = await tf.loadGraphModel('./model_tfjs/model.json');
}

// Stessa logica di predici()/esito in sportello_offline.py.
function predici(finestraFrame, finestra) {
  const feat = estendiSequenza(finestraFrame); // [T][744]
  const T = feat.length;
  const flat = new Float32Array(T * 744);
  feat.forEach((riga, t) => flat.set(riga, t * 744));

  const out = tf.tidy(() => {
    const input = tf.tensor3d(flat, [1, T, 744]);
    const pred  = modello.execute(input);
    return pred.reshape([SEGNI.length]).arraySync();
  });

  let idx1 = 0, idx2 = -1;
  for (let i = 1; i < out.length; i++) {
    if (out[i] > out[idx1]) { idx2 = idx1; idx1 = i; }
    else if (idx2 === -1 || out[i] > out[idx2]) { idx2 = i; }
  }

  const segno1 = SEGNI[idx1], segno2 = SEGNI[idx2];
  const conf1 = out[idx1], conf2 = out[idx2];
  const margine = conf1 - conf2;

  const isLettera = LETTERE.includes(segno1);
  const soglia    = isLettera ? SOGLIA_LETTERA : SOGLIA_PAROLA;

  let esito;
  if (conf1 < soglia) esito = 'conf_bassa';
  else if (COPPIE_AMBIGUE.has(coppiaKey(segno1, segno2)) && margine < MARGINE_AMBIGUE) esito = 'margine_basso';
  else esito = 'OK';

  return { segno1, conf1, segno2, conf2, margine, esito, isLettera, finestra };
}

function confermaSegno(candidatoNuovo) {
  if (candidatoNuovo === candidato) conferme++;
  else { candidato = candidatoNuovo; conferme = 1; }
  if (conferme >= CONFERME_MIN) {
    if (!frase.length || frase[frase.length - 1] !== candidato || manoAssente) {
      frase.push(candidato);
      if (frase.length > MAX_SEGNI) frase = frase.slice(-MAX_SEGNI);
    }
    manoAssente = false;
    candidato = ''; conferme = 0; bufferGrezzo = []; pausa = PAUSA_DOPO_CONFERMA;
    return true; // segno confermato, buffer azzerato
  }
  bufferGrezzo = bufferGrezzo.slice(SLIDE_STEP);
  return false;
}

// ── Voce (TTS) — legge ad alta voce le frasi segnate, per l'operatore ──
let vociPronte = [];
function caricaVoci() { vociPronte = speechSynthesis.getVoices(); }
if ('speechSynthesis' in window) {
  caricaVoci();
  speechSynthesis.onvoiceschanged = caricaVoci;
}
function parla(testo) {
  if (!('speechSynthesis' in window) || !testo) return;
  const u = new SpeechSynthesisUtterance(testo);
  u.lang = 'it-IT';
  const voceIt = vociPronte.find((v) => v.lang && v.lang.toLowerCase().startsWith('it'));
  if (voceIt) u.voice = voceIt;
  speechSynthesis.cancel(); // non accavallare piu' frasi
  speechSynthesis.speak(u);
}

// ── Correzione grammaticale offline (porto di _traduci_offline Python) ──
// Nessuna chiamata a Claude/internet: regole fisse (articoli, coniugazioni,
// negazione). Meno flessibile di Claude ma funziona sempre, anche offline.
function raggruppaLettere(parole) {
  const res = [];
  let i = 0;
  while (i < parole.length) {
    const p = parole[i];
    if (p.length === 1 && /^[a-zA-Z]$/.test(p)) {
      let w = p;
      while (i + 1 < parole.length && parole[i + 1].length === 1 && /^[a-zA-Z]$/.test(parole[i + 1])) {
        i++; w += parole[i];
      }
      res.push(w.toLowerCase());
    } else {
      res.push(p);
    }
    i++;
  }
  return res;
}

const _ARTICOLI = {
  casa: 'la', nome: 'il', giorno: 'il',
  mamma: 'la', papa: 'il', figlio: 'il',
  fratello: 'il', sorella: 'la',
};
const _POSSESSIVI = new Set(['mio', 'tuo']);
const _CONIUGAZIONI = {
  'io|aiutare': 'aiuto', 'tu|aiutare': 'aiuti',
  'io|capire': 'capisco', 'tu|capire': 'capisci',
  'io|mangiare': 'mangio', 'tu|mangiare': 'mangi',
};

function traduciOffline(parole) {
  parole = parole.map((p) => p.toLowerCase());
  const verbi = new Set(['aiutare', 'capire', 'mangiare']);
  const nomi = new Set(['casa', 'nome', 'giorno', 'mamma', 'papa', 'figlio', 'fratello', 'sorella']);
  const haNo = parole.includes('no');
  const soggetto = parole.find((p) => p === 'io' || p === 'tu') || null;
  const tokens = [];
  const skip = new Set();
  for (let i = 0; i < parole.length; i++) {
    if (skip.has(i)) continue;
    const p = parole[i];
    if (p === 'no') continue;
    if (verbi.has(p)) {
      if (haNo) tokens.push('non');
      tokens.push(_CONIUGAZIONI[`${soggetto}|${p}`] || p);
    } else if (nomi.has(p)) {
      const art = _ARTICOLI[p] || '';
      if (i + 1 < parole.length && _POSSESSIVI.has(parole[i + 1])) {
        const poss = parole[i + 1]; skip.add(i + 1);
        tokens.push(`${art} ${poss} ${p}`.trim());
      } else {
        tokens.push(`${art} ${p}`.trim());
      }
    } else if (p === 'si') {
      tokens.push('sì');
    } else {
      tokens.push(p);
    }
  }
  const frase = tokens.join(' ').replace(' ?', '?');
  return frase ? frase[0].toUpperCase() + frase.slice(1) : '';
}

function chiudiFrase() {
  if (frase.length) {
    const parole = raggruppaLettere(frase);
    const testo  = traduciOffline(parole);
    messaggi.push({ testo, mittente: 'io' });
    frase = [];
    renderChat();
    parla(testo);
  }
}

function inviaMessaggioOperatore(testo) {
  testo = testo.trim();
  if (!testo) return;
  messaggi.push({ testo, mittente: 'operatore' });
  renderChat();
}

function renderChat() {
  const chatEl = document.getElementById('chat');
  chatEl.innerHTML = '';
  for (const m of messaggi) {
    const b = document.createElement('div');
    b.className = 'bubble' + (m.mittente === 'operatore' ? ' operatore' : '');
    b.textContent = m.testo;
    chatEl.appendChild(b);
  }
  if (frase.length) {
    const b = document.createElement('div');
    b.className = 'bubble pending';
    b.textContent = frase.map((s) => s.toUpperCase()).join(' ');
    chatEl.appendChild(b);
  }
  chatEl.scrollTop = chatEl.scrollHeight;
}

document.getElementById('btn-pulisci').addEventListener('click', () => {
  frase = []; messaggi = []; candidato = ''; conferme = 0; bufferGrezzo = [];
  renderChat();
});

// ── Input testuale operatore (funziona ovunque, anche Safari) ──
const msgInput = document.getElementById('msg-input');
document.getElementById('btn-invia').addEventListener('click', () => {
  inviaMessaggioOperatore(msgInput.value);
  msgInput.value = '';
});
msgInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') {
    inviaMessaggioOperatore(msgInput.value);
    msgInput.value = '';
  }
});

// ── Trascina la camera PiP ovunque sullo schermo ──
const pipEl = document.getElementById('pip');
{
  // Converte da posizionamento "right" (CSS iniziale) a "left" (serve per
  // poter spostare liberamente in orizzontale via JS), senza far saltare
  // la posizione visiva.
  pipEl.style.left  = pipEl.offsetLeft + 'px';
  pipEl.style.right = 'auto';

  let trascinando = false;
  let startX = 0, startY = 0, startLeft = 0, startTop = 0;

  pipEl.addEventListener('pointerdown', (e) => {
    trascinando = true;
    startX = e.clientX; startY = e.clientY;
    startLeft = pipEl.offsetLeft; startTop = pipEl.offsetTop;
    pipEl.classList.add('trascinando');
    pipEl.setPointerCapture(e.pointerId);
  });

  pipEl.addEventListener('pointermove', (e) => {
    if (!trascinando) return;
    const stage    = document.getElementById('stage');
    const headerH  = document.getElementById('ui').clientHeight;
    const inputH   = document.getElementById('input-bar').clientHeight;

    let nuovoLeft = startLeft + (e.clientX - startX);
    let nuovoTop  = startTop  + (e.clientY - startY);
    nuovoLeft = Math.max(6, Math.min(nuovoLeft, stage.clientWidth  - pipEl.offsetWidth  - 6));
    nuovoTop  = Math.max(headerH + 6, Math.min(nuovoTop, stage.clientHeight - inputH - pipEl.offsetHeight - 6));
    pipEl.style.left = nuovoLeft + 'px';
    pipEl.style.top  = nuovoTop + 'px';
  });

  const fine = () => {
    if (!trascinando) return;
    trascinando = false;
    pipEl.classList.remove('trascinando');
  };
  pipEl.addEventListener('pointerup', fine);
  pipEl.addEventListener('pointercancel', fine);
}

// ── Microfono (STT) — Vosk offline (niente cloud, funziona senza internet) ──
// Sostituisce l'API SpeechRecognition del browser: quella manda l'audio ai
// server Google/Apple per la trascrizione e NON funziona offline. Vosk gira
// interamente in locale (WebAssembly + modello italiano vendorizzato).
const btnMic = document.getElementById('btn-mic');
if (window.Vosk) {
  btnMic.hidden = false;
  let voskModel = null, ascolto = false;
  let audioContext = null, recognizerNode = null, micStream = null, recognizer = null;

  async function caricaVosk() {
    if (voskModel) return voskModel;
    btnMic.disabled = true;
    btnMic.title = 'Caricamento riconoscimento vocale...';
    voskModel = await window.Vosk.createModel('./vendor/vosk/model.tar.gz');
    btnMic.disabled = false;
    btnMic.title = '';
    return voskModel;
  }

  async function avviaAscolto() {
    const model = await caricaVosk();
    recognizer = new model.KaldiRecognizer(16000);
    recognizer.setWords(false);
    recognizer.on('result', (msg) => {
      const testo = (msg.result && msg.result.text) ? msg.result.text.trim() : '';
      if (testo) inviaMessaggioOperatore(testo);
    });

    micStream = await navigator.mediaDevices.getUserMedia({
      video: false,
      audio: { echoCancellation: true, noiseSuppression: true, channelCount: 1, sampleRate: 16000 },
    });
    audioContext = new AudioContext({ sampleRate: 16000 });
    recognizerNode = audioContext.createScriptProcessor(4096, 1, 1);
    recognizerNode.onaudioprocess = (event) => {
      try { recognizer.acceptWaveform(event.inputBuffer); } catch (err) { console.error('Vosk acceptWaveform:', err); }
    };
    const source = audioContext.createMediaStreamSource(micStream);
    source.connect(recognizerNode);
    recognizerNode.connect(audioContext.destination);
  }

  function fermaAscolto() {
    if (micStream) micStream.getTracks().forEach((t) => t.stop());
    if (recognizerNode) recognizerNode.disconnect();
    if (audioContext) audioContext.close();
    if (recognizer) recognizer.remove();
    micStream = audioContext = recognizerNode = recognizer = null;
  }

  btnMic.addEventListener('click', async () => {
    if (ascolto) {
      ascolto = false; btnMic.classList.remove('attivo');
      fermaAscolto();
      return;
    }
    ascolto = true; btnMic.classList.add('attivo');
    try {
      await avviaAscolto();
    } catch (err) {
      console.error('Avvio microfono fallito:', err);
      ascolto = false; btnMic.classList.remove('attivo');
    }
  });
}

// ── Estrazione keypoints — stesso ordine di estrai_keypoints() Python ──
// lh(63) + rh(63) + pose(24) + face(210) = 360, formato raccolta_dati_5.py
function estraiKeypoints(handRes, faceRes, poseRes) {
  let lh = new Array(63).fill(0);
  let rh = new Array(63).fill(0);

  if (handRes && handRes.landmarks.length) {
    handRes.handedness.forEach((h, idx) => {
      const label = h[0].categoryName; // 'Left' | 'Right'
      const flat  = handRes.landmarks[idx].flatMap(p => [p.x, p.y, p.z]);
      if (label === 'Left') lh = flat; else rh = flat;
    });
  }

  let pose = new Array(POSE_IDX.length * 3).fill(0);
  if (poseRes && poseRes.landmarks.length) {
    const lm = poseRes.landmarks[0];
    pose = POSE_IDX.flatMap(i => [lm[i].x, lm[i].y, lm[i].z]);
  }

  let face = new Array(FACE_IDX.length * 3).fill(0);
  if (faceRes && faceRes.faceLandmarks.length) {
    const lm = faceRes.faceLandmarks[0];
    face = FACE_IDX.flatMap(i => [lm[i].x, lm[i].y, lm[i].z]);
  }

  return [...lh, ...rh, ...pose, ...face];
}

// ── Loop principale ─────────────────────────────────────────
// Il video si ridisegna ad ogni frame (fluido), ma le 3 rilevazioni ML
// (mani/viso/corpo) girano solo ogni INTERVALLO_DETECT ms: farle girare
// a piena velocita' del refresh dello schermo (fino a 60/sec) blocca il
// thread unico del browser sui telefoni meno potenti (rilevato su iPhone
// 12 -- "si blocca, non fluido"). 10 rilevazioni al secondo bastano per
// il riconoscimento e lasciano il resto dell'app reattivo.
let ultimoTempo = -1;
let ultimoDetect = 0;
const INTERVALLO_DETECT = 100;
// Ultimi risultati validi: disegnati ad OGNI frame (fluido, niente
// scheletro che sparisce e riappare), anche nei frame in cui non gira
// una nuova rilevazione.
let ultimoHandRes = null, ultimoFaceRes = null, ultimoPoseRes = null;

function loop() {
  requestAnimationFrame(loop);
  if (video.readyState < 2) return;

  const now = performance.now();
  if (now === ultimoTempo) return;
  ultimoTempo = now;

  // "Cuoce" il mirror nei pixel, come cv2.flip(frame,1) in Python —
  // cosi' la classificazione mano sinistra/destra di MediaPipe combacia.
  workCtx.save();
  workCtx.scale(-1, 1);
  workCtx.drawImage(video, -workCanvas.width, 0, workCanvas.width, workCanvas.height);
  workCtx.restore();
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  ctx.drawImage(workCanvas, 0, 0, overlay.width, overlay.height);

  let novitaRilevazione = false;
  if (now - ultimoDetect >= INTERVALLO_DETECT) {
    ultimoDetect = now;
    ultimoHandRes = handLandmarker.detectForVideo(workCanvas, now);
    ultimoFaceRes = faceLandmarker.detectForVideo(workCanvas, now);
    ultimoPoseRes = poseLandmarker.detectForVideo(workCanvas, now);
    novitaRilevazione = true;
  }
  const handRes = ultimoHandRes, faceRes = ultimoFaceRes, poseRes = ultimoPoseRes;
  if (!handRes || !faceRes || !poseRes) return;

  if (poseRes.landmarks[0]) disegnaPose(poseRes.landmarks[0]);
  if (faceRes.faceLandmarks[0]) disegnaViso(faceRes.faceLandmarks[0]);
  handRes.landmarks.forEach((lm, idx) => {
    const label = handRes.handedness[idx][0].categoryName;
    disegnaMano(lm, label === 'Left' ? '#00ff88' : '#ff5050');
  });

  // Riconoscimento (estrazione keypoint + predizione Conformer) solo sui
  // dati appena rilevati -- rifarlo su dati identici ad ogni frame
  // sprecherebbe esattamente il calcolo che si voleva ridurre sopra.
  if (!novitaRilevazione) return;

  const keypoints = estraiKeypoints(handRes, faceRes, poseRes);
  const manoVista = handRes.landmarks.length > 0;

  let debug = '';
  if (pausa > 0) {
    pausa--;
    setStato(`pausa (${pausa})`, 'wait');
    return;
  }
  if (manoVista) {
    manoPersa = 0;
    bufferGrezzo.push(keypoints);
    if (bufferGrezzo.length > SEQ_LEN_PAROLA) bufferGrezzo = bufferGrezzo.slice(-SEQ_LEN_PAROLA);

    let letteraInCorso = false;

    // 1) Finestra corta: solo lettere
    if (bufferGrezzo.length >= SEQ_LEN_LETTERA) {
      const r5 = predici(bufferGrezzo.slice(-SEQ_LEN_LETTERA), 5);
      debug = `${r5.segno1}(${r5.conf1.toFixed(2)}) ${r5.esito}`;
      if (r5.isLettera && r5.esito === 'OK') {
        letteraInCorso = true;
        confermaSegno(r5.segno1);
      }
    }

    // 2) Finestra lunga: parole (e lettere di riserva)
    if (!letteraInCorso && bufferGrezzo.length >= SEQ_LEN_PAROLA) {
      const r8 = predici(bufferGrezzo.slice(-SEQ_LEN_PAROLA), 8);
      debug = `${r8.segno1}(${r8.conf1.toFixed(2)}) ${r8.esito}`;
      if (r8.esito === 'OK') {
        confermaSegno(r8.segno1);
      } else if (r8.esito === 'margine_basso') {
        bufferGrezzo = bufferGrezzo.slice(SLIDE_STEP);
      } else {
        candidato = ''; conferme = 0; bufferGrezzo = bufferGrezzo.slice(SLIDE_STEP);
      }
    }

    frameSenzaMani = 0;
    renderChat();
  } else if (bufferGrezzo.length && manoPersa < TOLLERANZA_PERSA) {
    // Tolleranza: mano persa un istante (movimento veloce, vicino a
    // spalla/testa) — aspetta invece di azzerare il buffer
    manoPersa++;
    frameSenzaMani++;
    if (frameSenzaMani === PAUSA_FRASE) chiudiFrase();
  } else {
    bufferGrezzo = []; candidato = ''; conferme = 0; manoAssente = true; manoPersa = 0;
    frameSenzaMani++;
    if (frameSenzaMani === PAUSA_FRASE) chiudiFrase();
  }

  setStato(manoVista ? `mano (${handRes.landmarks.length}) ${debug}` : 'in attesa mano...', manoVista ? 'ok' : 'wait');
}

async function main() {
  try {
    await initCamera();
    await Promise.all([initModelli(), initModello()]);
    loadingEl.style.display = 'none';
    loop();
  } catch (err) {
    setStato('errore', 'err');
    loadingEl.textContent = 'Errore: ' + err.message;
    console.error(err);
  }
}

main();
