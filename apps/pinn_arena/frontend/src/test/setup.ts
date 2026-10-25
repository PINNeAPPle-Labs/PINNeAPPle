import '@testing-library/jest-dom'

// Node >= 22 ships its own global `localStorage`, which is unusable unless Node
// runs with --localstorage-file and shadows jsdom's implementation. Anything
// touching localStorage (the axios auth interceptor, authStore) then throws.
// Install an in-memory Storage when the global one doesn't work.
function storageWorks(): boolean {
  try {
    return typeof globalThis.localStorage?.getItem === 'function'
  } catch {
    return false
  }
}

if (!storageWorks()) {
  const store = new Map<string, string>()
  const memoryStorage: Storage = {
    get length() { return store.size },
    clear: () => store.clear(),
    getItem: (key) => (store.has(key) ? store.get(key)! : null),
    key: (index) => Array.from(store.keys())[index] ?? null,
    removeItem: (key) => { store.delete(key) },
    setItem: (key, value) => { store.set(key, String(value)) },
  }
  Object.defineProperty(globalThis, 'localStorage', { value: memoryStorage, configurable: true, writable: true })
}
