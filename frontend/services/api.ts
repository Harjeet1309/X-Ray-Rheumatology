/**
 * Typed API Service Client for the FastAPI Rheumatological X-ray Analysis Backend.
 */

export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface QualityMetadata {
  valid: boolean;
  width: number;
  height: number;
  aspect_ratio: number;
  mean_intensity: number;
  std_intensity: number;
  dynamic_range: number;
  sharpness_score: number;
  quality_warnings: string[];
}

export interface UploadResponse {
  analysis_id: string;
  filename: string;
  analysis_status: string;
  upload_timestamp: string;
  quality_metadata: QualityMetadata;
  preview_url: string;
}

export async function uploadXRayFile(
  file: File,
  token?: string | null
): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append("file", file);
  const headers: Record<string, string> = {};
  if (token) headers["Authorization"] = `Bearer ${token}`;

  const res = await fetch(`${API_BASE_URL}/api/analysis/upload`, {
    method: "POST",
    headers,
    body: formData,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Failed to upload X-ray radiograph");
  }
  return res.json();
}

export async function loadSampleXRay(
  caseType: "ra_positive" | "normal" = "ra_positive",
  token?: string | null
): Promise<UploadResponse> {
  const headers: Record<string, string> = {};
  if (token) headers["Authorization"] = `Bearer ${token}`;

  const res = await fetch(
    `${API_BASE_URL}/api/analysis/sample?case_type=${caseType}`,
    {
      method: "POST",
      headers,
    }
  );
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Failed to load sample RAM-H1200 X-ray");
  }
  return res.json();
}

export async function runAnalysisById(analysisId: string): Promise<any> {
  const res = await fetch(`${API_BASE_URL}/api/analysis/${analysisId}/run`, {
    method: "POST",
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "AI inference execution failed");
  }
  return res.json();
}

export async function fetchAnalysisDetail(analysisId: string): Promise<any> {
  const res = await fetch(`${API_BASE_URL}/api/analysis/${analysisId}`);
  if (!res.ok) throw new Error("Could not load analysis details");
  return res.json();
}

export async function fetchAnalysisHistory(): Promise<any> {
  const res = await fetch(`${API_BASE_URL}/api/analysis/history`);
  if (!res.ok) throw new Error("Could not load analysis history");
  return res.json();
}

export async function deleteAnalysisById(analysisId: string): Promise<any> {
  const res = await fetch(`${API_BASE_URL}/api/analysis/${analysisId}`, {
    method: "DELETE",
  });
  if (!res.ok) throw new Error("Failed to delete analysis record");
  return res.json();
}

export async function fetchModelsAndAudit(): Promise<any> {
  const res = await fetch(`${API_BASE_URL}/api/models`);
  if (!res.ok) throw new Error("Could not load model registry");
  return res.json();
}

export async function registerUserAccount(
  name: string,
  email: string,
  password: string
): Promise<any> {
  const res = await fetch(`${API_BASE_URL}/api/auth/register`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, email, password }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Registration failed");
  }
  return res.json();
}

export async function loginUserAccount(
  email: string,
  password: string
): Promise<any> {
  const res = await fetch(`${API_BASE_URL}/api/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Authentication failed");
  }
  return res.json();
}
