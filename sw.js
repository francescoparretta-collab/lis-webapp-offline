// Service worker OFFLINE — mette in cache TUTTO al primo avvio (pagina,
// librerie MediaPipe/TensorFlow.js, modelli), cosi' l'app funziona anche
// senza connessione dopo la prima apertura.
const CACHE = 'lis-offline-v14';

const FILE_ASSETS = [
  './', './index.html', './app.js', './features.js', './manifest.json',
  './icons/icon.svg',
  './vendor/tfjs/tf.min.js',
  './vendor/vosk/vosk.js',
  './vendor/vosk/model.tar.gz',
  './vendor/tasks-vision/vision_bundle.mjs',
  './vendor/tasks-vision/wasm/vision_wasm_internal.js',
  './vendor/tasks-vision/wasm/vision_wasm_internal.wasm',
  './vendor/tasks-vision/wasm/vision_wasm_module_internal.js',
  './vendor/tasks-vision/wasm/vision_wasm_module_internal.wasm',
  './vendor/tasks-vision/wasm/vision_wasm_nosimd_internal.js',
  './vendor/tasks-vision/wasm/vision_wasm_nosimd_internal.wasm',
  './vendor/models/hand_landmarker.task',
  './vendor/models/face_landmarker.task',
  './vendor/models/pose_landmarker_lite.task',
  './model_tfjs/model.json',
  './model_tfjs/group1-shard1of4.bin',
  './model_tfjs/group1-shard2of4.bin',
  './model_tfjs/group1-shard3of4.bin',
  './model_tfjs/group1-shard4of4.bin',
];

self.addEventListener('install', (event) => {
  event.waitUntil(caches.open(CACHE).then((cache) => cache.addAll(FILE_ASSETS)));
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
  );
  self.clients.claim();
});

// Cache-first: una volta scaricato tutto, non serve piu' la rete.
self.addEventListener('fetch', (event) => {
  if (event.request.method !== 'GET') return;
  event.respondWith(
    caches.match(event.request).then((cached) => cached || fetch(event.request))
  );
});
