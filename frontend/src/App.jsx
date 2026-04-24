import { useState } from 'react'
import AuthPage from './components/AuthPage'
import FeedPage from './components/FeedPage'
import { getStoredAuth } from './services/api'

export default function App() {
  const stored = getStoredAuth()
  const [token, setToken] = useState(stored.token)
  const [user, setUser] = useState(stored.user)

  if (!token || !user) {
    return <AuthPage onAuthenticated={(newToken, newUser) => { setToken(newToken); setUser(newUser) }} />
  }

  return <FeedPage token={token} user={user} onLogout={() => { setToken(null); setUser(null) }} />
}
