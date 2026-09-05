/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_BASE_URL?: string
  /** Phase 7 - shared-secret API key sent as the X-API-Key header on every
   * request (see src/api/client.ts). Must match the backend's
   * RISK_API_API_KEY. Never commit a real value - see .env.example. */
  readonly VITE_API_KEY?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
