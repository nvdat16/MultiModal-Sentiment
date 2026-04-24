import { getImageUrl } from '../services/api'
import { avatarUrl, sentimentClass } from '../utils/format'

export default function PostCard({ post }) {
  const image = getImageUrl(post.image_url)
  const confidence = post.confidence ?? 0

  return (
    <article className="post-card">
      <header className="post-author">
        <img src={avatarUrl(post.user?.name)} alt="avatar" />
        <div>
          <b>{post.user?.name}</b>
          <span>{new Date(post.created_at).toLocaleString('vi-VN')} · 🌎</span>
        </div>
      </header>

      {post.content && <p className="post-content">{post.content}</p>}

      <div className="post-media">
        {image && <img src={image} alt="post" />}
        <div className={image ? 'sentiment-pill floating' : 'sentiment-pill'}>
          <span className="live-dot" />
          <b>AI Sentiment:</b>
          <span className={sentimentClass(post.sentiment)}>{post.sentiment || '—'} ({confidence}%)</span>
        </div>
      </div>

      <div className="prob-grid">
        <span>Positive: <b className="sent-positive">{post.positive ?? '—'}%</b></span>
        <span>Neutral: <b className="sent-neutral">{post.neutral ?? '—'}%</b></span>
        <span>Negative: <b className="sent-negative">{post.negative ?? '—'}%</b></span>
      </div>

      <footer className="post-actions">
        <button>👍 Thích</button>
        <button>💬 Bình luận</button>
      </footer>
    </article>
  )
}
