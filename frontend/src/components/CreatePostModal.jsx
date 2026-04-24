import { useEffect, useState } from 'react'
import { createPost } from '../services/api'
import { avatarUrl } from '../utils/format'

export default function CreatePostModal({ user, open, onClose, token, onCreated }) {
  const [text, setText] = useState('')
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState('')
  const [message, setMessage] = useState('')
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    if (!file) {
      setPreview('')
      return
    }
    const url = URL.createObjectURL(file)
    setPreview(url)
    return () => URL.revokeObjectURL(url)
  }, [file])

  if (!open) return null

  async function handleSubmit() {
    if (!text.trim() && !file) {
      setMessage('Vui lòng nhập nội dung hoặc chọn ảnh.')
      return
    }

    const formData = new FormData()
    if (text.trim()) formData.append('text', text.trim())
    if (file) formData.append('file', file)

    setLoading(true)
    setMessage('Đang gọi model dự đoán sentiment...')
    try {
      await createPost(token, formData)
      setText('')
      setFile(null)
      setMessage('')
      onCreated()
      onClose()
    } catch (error) {
      setMessage(error.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="modal-backdrop" onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
      <section className="post-modal">
        <header className="modal-head">
          <span />
          <h2>Tạo bài viết</h2>
          <button onClick={onClose} className="icon-btn" type="button">✕</button>
        </header>
        <div className="modal-body">
          <div className="user-row">
            <img src={avatarUrl(user?.name)} alt="avatar" />
            <div>
              <b>{user?.name}</b>
              <span>🌎 Công khai</span>
            </div>
          </div>

          <textarea
            value={text}
            onChange={(event) => setText(event.target.value)}
            placeholder="Bạn đang nghĩ gì thế?"
            autoFocus
          />

          {preview && <img className="image-preview" src={preview} alt="preview" />}

          <div className="add-row">
            <b>Thêm vào bài viết của bạn</b>
            <label className="image-picker">🖼️
              <input type="file" accept="image/*" onChange={(event) => setFile(event.target.files?.[0] || null)} />
            </label>
          </div>

          <button onClick={handleSubmit} disabled={loading} className="primary-btn full-btn" type="button">
            {loading ? 'Đang phân tích...' : 'Đăng & phân tích sentiment'}
          </button>
          <p className="post-message">{message}</p>
        </div>
      </section>
    </div>
  )
}
