const base = (import.meta.env.VITE_API_BASE_URL || "/api").replace(/\/$/, "");

export async function api(path, options = {}) {
  const response = await fetch(`${base}${path}`, options);
  if (response.status === 204) return null;
  const body = await response.json();
  if (!response.ok) {
    const message =
      typeof body.detail === "string"
        ? body.detail
        : "The request could not be completed.";
    throw new Error(message);
  }
  return body;
}
