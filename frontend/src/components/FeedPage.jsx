import { useEffect, useMemo, useState } from 'react'
import { clearAuth, getPosts } from '../services/api'
import { avatarUrl } from '../utils/format'
import CreatePostModal from './CreatePostModal'
import PostCard from './PostCard'

export default function FeedPage({ token, user, onLogout }) {
  const [posts, setPosts] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [modalOpen, setModalOpen] = useState(false)

  const avatar = useMemo(() => avatarUrl(user?.name), [user])

  async function loadFeed() {
    setLoading(true)
    setError('')
    try {
      const data = await getPosts(token)
      setPosts(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadFeed()
  }, [])

  function handleLogout() {
    clearAuth()
    onLogout()
  }

  return (
    <>
      <nav className="top-nav">
        <div className="nav-inner">
          <div className="brand">Twitter.</div>
          <div className="search-box">🔎 <input placeholder="Tìm kiếm bài viết..." /></div>
          <div className="nav-user">
            <button onClick={handleLogout}>Đăng xuất</button>
            <img src={avatar} alt="avatar" />
          </div>
        </div>
      </nav>

      <main className="layout">
        <aside className="left-sidebar">
          <div className="side-item"><img src={avatar} alt="avatar" /><b>{user?.name}</b></div>
          <div className="side-item"><span>📚</span> Thư viện bài viết</div>
        </aside>

        <section className="feed-column">
          <button className="composer" onClick={() => setModalOpen(true)} type="button">
            <img src={avatar} alt="avatar" />
            <span>Bạn đang nghĩ gì thế?</span>
          </button>

          {loading && <div className="empty-card">Đang tải bài viết...</div>}
          {error && <div className="error-card">{error}</div>}
          {!loading && !error && posts.length === 0 && <div className="empty-card">Chưa có bài viết nào.</div>}
          {!loading && !error && posts.map((post) => <PostCard key={post.id} post={post} token={token} />)}
        </section>

        <aside className="right-sidebar">
          <div className="info-card">
            <h4>Thông tin đồ án</h4>
            <b>Đồ án 2</b>
            <span>Họ tên: Nguyễn Văn Đạt</span>
            <span>MSSV: 12423061</span>
            <span>Lớp: 12423TN</span>
          </div>
        </aside>
      </main>

      <CreatePostModal
        user={user}
        open={modalOpen}
        onClose={() => setModalOpen(false)}
        token={token}
        onCreated={loadFeed}
      />
    </>
  )
}
