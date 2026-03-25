import type {
  CollateralResponse,
  DatasetId,
  DatasetsResponse,
  ExamplesResponse,
  GroupSpec,
  HoldoutCurvesResponse,
  OptionsResponse,
  PartitionComparisonsResponse,
  Phase,
} from "./types";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
    ...init,
  });

  if (!response.ok) {
    let detail = response.statusText;
    try {
      const payload = await response.json();
      detail = payload.detail ?? JSON.stringify(payload);
    } catch {
      detail = await response.text();
    }
    throw new Error(detail || "Request failed");
  }

  return (await response.json()) as T;
}

export function getApiBaseUrl(): string {
  return API_BASE_URL;
}

export function getDatasets(): Promise<DatasetsResponse> {
  return request<DatasetsResponse>("/api/flir-analysis/datasets");
}

export function getOptions(dataset: DatasetId): Promise<OptionsResponse> {
  return request<OptionsResponse>(`/api/flir-analysis/options?dataset=${encodeURIComponent(dataset)}`);
}

export function getHoldoutCurves(payload: {
  dataset: DatasetId;
  phase: Phase;
  groups: GroupSpec[];
  thresholds?: number[];
}): Promise<HoldoutCurvesResponse> {
  return request<HoldoutCurvesResponse>("/api/flir-analysis/holdout-curves", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function getCollateral(payload: {
  dataset: DatasetId;
  phase: Phase;
  groups: GroupSpec[];
  tau: number;
}): Promise<CollateralResponse> {
  return request<CollateralResponse>("/api/flir-analysis/collateral", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function getPartitionComparisons(payload: {
  dataset: DatasetId;
  phase: Phase;
  groups: GroupSpec[];
  tau: number;
  include_zero_counts?: boolean;
}): Promise<PartitionComparisonsResponse> {
  return request<PartitionComparisonsResponse>("/api/flir-analysis/partition-comparisons", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function getExamples(payload: {
  dataset: DatasetId;
  phase: Phase;
  groups: GroupSpec[];
  tau: number;
  example_count?: number;
}): Promise<ExamplesResponse> {
  return request<ExamplesResponse>("/api/flir-analysis/examples", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}
