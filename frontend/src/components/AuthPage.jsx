import { useState } from 'react'
import { login, register, saveAuth } from '../services/api'

export default function AuthPage({ onAuthenticated }) {
  const [mode, setMode] = useState('login')
  const [form, setForm] = useState({ name: '', email: '', password: '' })
  const [message, setMessage] = useState('')
  const [loading, setLoading] = useState(false)

  const isRegister = mode === 'register'

  async function handleSubmit(event) {
    event.preventDefault()
    setLoading(true)
    setMessage('Đang xử lý...')

    try {
      const payload = isRegister
        ? { name: form.name, email: form.email, password: form.password }
        : { email: form.email, password: form.password }
      const data = isRegister ? await register(payload) : await login(payload)
      saveAuth(data.access_token, data.user)
      onAuthenticated(data.access_token, data.user)
    } catch (error) {
      setMessage(error.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className="auth-page">
      <section className="auth-hero">
        <div className="brand-xl">TASTE.</div>
        <h1>Social feed tích hợp Multimodal Sentiment AI</h1>
        <p>Đăng bài bằng text và hình ảnh, backend FastAPI gọi model sentiment và lưu tài khoản, bài đăng vào MySQL.</p>
      </section>

      <section className="auth-card">
        <div className="auth-tabs">
          <button className={mode === 'login' ? 'active' : ''} onClick={() => setMode('login')} type="button">Đăng nhập</button>
          <button className={mode === 'register' ? 'active' : ''} onClick={() => setMode('register')} type="button">Đăng ký</button>
        </div>

        <form onSubmit={handleSubmit} className="auth-form">
          {isRegister && (
            <input
              value={form.name}
              onChange={(event) => setForm({ ...form, name: event.target.value })}
              placeholder="Họ tên"
              required
            />
          )}
          <input
            value={form.email}
            onChange={(event) => setForm({ ...form, email: event.target.value })}
            type="email"
            placeholder="Email"
            required
          />
          <input
            value={form.password}
            onChange={(event) => setForm({ ...form, password: event.target.value })}
            type="password"
            placeholder="Mật khẩu"
            required
          />
          <button disabled={loading} className="primary-btn">{loading ? 'Đang xử lý...' : isRegister ? 'Đăng ký' : 'Đăng nhập'}</button>
          <p className="form-message">{message}</p>
        </form>
      </section>
    </main>
  )
}
