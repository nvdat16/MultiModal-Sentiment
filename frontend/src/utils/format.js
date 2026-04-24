export function avatarUrl(name) {
  return `https://ui-avatars.com/api/?name=${encodeURIComponent(name || 'User')}&background=EA580C&color=fff`
}

export function sentimentClass(sentiment = '') {
  const value = sentiment.toLowerCase()
  if (value.includes('positive')) return 'sent-positive'
  if (value.includes('negative')) return 'sent-negative'
  return 'sent-neutral'
}
