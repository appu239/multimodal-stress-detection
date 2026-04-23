// src/utils/api.js

// ⚠️ TO DEPLOY: Replace the PRODUCTION_URL below with your actual backend URL
// e.g., your Render URL: "https://your-app.onrender.com"
// e.g., your HuggingFace Space URL: "https://username-spacename.hf.space"
const PRODUCTION_URL = "https://YOUR_BACKEND_URL_HERE";
const LOCAL_URL = "http://10.44.181.233:5000"; // Used when running locally

export const API_BASE_URL = process.env.NODE_ENV === "production" ? PRODUCTION_URL : LOCAL_URL;

export const getApiUrl = (endpoint) => {
  const cleanEndpoint = endpoint.startsWith("/") ? endpoint : `/${endpoint}`;
  return `${API_BASE_URL}${cleanEndpoint}`;
};
