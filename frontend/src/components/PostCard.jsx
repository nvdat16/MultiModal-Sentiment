import { useEffect, useState } from 'react'
import { createComment, getImageUrl, getPostComments } from '../services/api'
import { avatarUrl, sentimentClass } from '../utils/format'

export default function PostCard({ post, token }) {
  const [comments, setComments] = useState([])
  const [showComments, setShowComments] = useState(false)
  const [showAllComments, setShowAllComments] = useState(false)
  const [commentText, setCommentText] = useState('')
  const [commentFile, setCommentFile] = useState(null)
  const [commentLoading, setCommentLoading] = useState(false)
  const [commentError, setCommentError] = useState('')
  const [loadingComments, setLoadingComments] = useState(false)

  const image = getImageUrl(post.image_url)
  const confidence = post.confidence ?? 0

  useEffect(() => {
    if (!showComments) return

    let active = true
    async function loadComments() {
      setLoadingComments(true)
      setCommentError('')
      try {
        const data = await getPostComments(token, post.id, !showAllComments)
        if (active) setComments(data)
      } catch (error) {
        if (active) setCommentError(error.message)
      } finally {
        if (active) setLoadingComments(false)
      }
    }

    loadComments()
    return () => {
      active = false
    }
  }, [showComments, post.id, token, showAllComments])

  async function handleSubmitComment(event) {
    event.preventDefault()
    if (!commentText.trim()) {
      setCommentError('Vui lòng nhập nội dung bình luận.')
      return
    }

    const formData = new FormData()
    formData.append('content', commentText.trim())
    if (commentFile) formData.append('file', commentFile)

    setCommentLoading(true)
    setCommentError('')
    try {
      await createComment(token, post.id, formData)
      setCommentText('')
      setCommentFile(null)
      const data = await getPostComments(token, post.id, !showAllComments)
      setComments(data)
      setShowComments(true)
    } catch (error) {
      setCommentError(error.message)
    } finally {
      setCommentLoading(false)
    }
  }

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
        <button type="button">👍 Thích</button>
        <button type="button" onClick={() => setShowComments(true)}>💬 Bình luận</button>
      </footer>

      {showComments && (
        <div
          className="comment-backdrop"
          onMouseDown={(event) => event.target === event.currentTarget && setShowComments(false)}
        >
          <section className="comment-modal" role="dialog" aria-modal="true" aria-labelledby={`comments-title-${post.id}`}>
            <header className="comment-modal-head">
              <div>
                <p className="comment-modal-kicker">Bài viết của</p>
                <h2 id={`comments-title-${post.id}`}>{post.user?.name || 'Người tham gia ẩn danh'}</h2>
              </div>
              <button type="button" className="icon-btn" onClick={() => setShowComments(false)}>✕</button>
            </header>

            <div className="comment-toolbar">
              <button
                type="button"
                className="comment-toggle"
                onClick={() => setShowAllComments((value) => !value)}
              >
                {showAllComments ? 'Hiển thị positive + neutral' : 'Hiển thị tất cả bình luận'}
              </button>
            </div>

            <div className="comment-modal-scroll">
              {loadingComments ? (
                <div className="comment-empty">Đang tải bình luận...</div>
              ) : comments.length === 0 ? (
                <div className="comment-empty">Chưa có bình luận nào phù hợp.</div>
              ) : (
                <div className="comment-list">
                  {comments.map((comment) => (
                    <article key={comment.id} className="comment-item">
                      <img src={avatarUrl(comment.user?.name)} alt="avatar" />
                      <div>
                        <div className="comment-head">
                          <b>{comment.user?.name}</b>
                          <span>{new Date(comment.created_at).toLocaleString('vi-VN')}</span>
                        </div>
                        <p>{comment.content}</p>
                        <small className={sentimentClass(comment.sentiment)}>
                          AI: {comment.sentiment || '—'} ({comment.confidence ?? 0}%)
                        </small>
                      </div>
                    </article>
                  ))}
                </div>
              )}
            </div>

            <form className="comment-form" onSubmit={handleSubmitComment}>
              <textarea
                value={commentText}
                onChange={(event) => setCommentText(event.target.value)}
                placeholder="Viết bình luận..."
                rows={3}
              />
              <div className="comment-tools">
                <label className="comment-file">
                  📎 Ảnh
                  <input type="file" accept="image/*" onChange={(event) => setCommentFile(event.target.files?.[0] || null)} />
                </label>
                <button className="primary-btn" type="submit" disabled={commentLoading}>
                  {commentLoading ? 'Đang phân tích...' : 'Gửi bình luận'}
                </button>
              </div>
              <p className="post-message">{commentError}</p>
            </form>
          </section>
        </div>
      )}
    </article>
  )
}
