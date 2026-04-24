const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000'

async function parseResponse(response) {
  const data = await response.json().catch(() => ({}))
  if (!response.ok) {
    throw new Error(data.detail || 'Có lỗi xảy ra')
  }
  return data
}

export function getStoredAuth() {
  return {
    token: localStorage.getItem('token'),
    user: JSON.parse(localStorage.getItem('user') || 'null'),
  }
}

export function saveAuth(token, user) {
  localStorage.setItem('token', token)
  localStorage.setItem('user', JSON.stringify(user))
}

export function clearAuth() {
  localStorage.removeItem('token')
  localStorage.removeItem('user')
}

export async function login(payload) {
  const response = await fetch(`${API_BASE}/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  return parseResponse(response)
}

export async function register(payload) {
  const response = await fetch(`${API_BASE}/auth/register`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  return parseResponse(response)
}

export async function getPosts(token) {
  const response = await fetch(`${API_BASE}/posts`, {
    headers: { Authorization: `Bearer ${token}` },
  })
  return parseResponse(response)
}

export async function createPost(token, formData) {
  const response = await fetch(`${API_BASE}/posts`, {
    method: 'POST',
    headers: { Authorization: `Bearer ${token}` },
    body: formData,
  })
  return parseResponse(response)
}

export function getImageUrl(imageUrl) {
  if (!imageUrl) return ''
  if (imageUrl.startsWith('http')) return imageUrl
  return `${API_BASE}${imageUrl}`
}
