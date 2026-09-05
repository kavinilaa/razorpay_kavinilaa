import { apiRequest } from './client'
import type { HealthResponse } from './types'

/** GET /api/v1/health - see backend/api/health.py. */
export async function fetchHealth() {
  return apiRequest<HealthResponse>('/api/v1/health')
}
