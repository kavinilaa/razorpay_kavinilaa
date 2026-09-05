import { apiRequest } from './client'
import type { ModelInfoResponse } from './types'

/** GET /api/v1/model-info - see backend/api/model_info.py. Can 503 if
 * models/model_metadata.json is missing/corrupt; apiRequest() surfaces that
 * as an `http_error` outcome rather than throwing. */
export async function fetchModelInfo() {
  return apiRequest<ModelInfoResponse>('/api/v1/model-info')
}
