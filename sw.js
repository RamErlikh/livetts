const CACHE_NAME = 'tts-translator-v1';
const urlsToCache = [
  '/',
  '/index.html',
  '/overlay.html',
  '/style.css',
  '/script.js',
  '/script-whisper.js',
  '/manifest.json'
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then((cache) => {
        return cache.addAll(urlsToCache);
      })
  );
});

self.addEventListener('fetch', (event) => {
  event.respondWith(
    caches.match(event.request)
      .then((response) => {
        // Return cached version or fetch from network
        return response || fetch(event.request);
      }
    )
  );
}); 