const BASE_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:5000";

async function request(path, options = {}) {
  let response;
  try {
    response = await fetch(`${BASE_URL}${path}`, {
      headers: { "Content-Type": "application/json" },
      ...options,
    });
  } catch (err) {
    throw new Error("OFFLINE");
  }

  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.message || `Request failed (${response.status})`);
  }

  return response.json();
}

export const getHealth = () => request("/api/health");

export const getDatasets = () => request("/api/datasets");

export const getEvaluation = () => request("/api/evaluation");

export const getBenchmarks = () => request("/api/benchmarks");

export const predict = (payload) =>
  request("/api/predict", {
    method: "POST",
    body: JSON.stringify(payload),
  });