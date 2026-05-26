const db = require('../config/db');

// ── GET NOTIFICATIONS ──────────────────────
exports.getNotifications = async (req, res) => {
  try {
    const result = await db.query(
      `SELECT * FROM notifications
       WHERE user_id=$1
       ORDER BY created_at DESC`,
      [req.user.id]
    );
    res.json(result.rows);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
};

// ── MARK NOTIFICATION AS READ ─────────────
exports.markAsRead = async (req, res) => {
  try {
    await db.query(
      'UPDATE notifications SET is_read=TRUE WHERE id=$1 AND user_id=$2',
      [req.params.id, req.user.id]
    );
    res.json({ message: 'Notification marked as read.' });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
};

// ── CLEAR ALL NOTIFICATIONS ───────────────
exports.clearAll = async (req, res) => {
  try {
    await db.query('DELETE FROM notifications WHERE user_id=$1', [req.user.id]);
    res.json({ message: 'All notifications cleared.' });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
};

// ── GET GENERATION HISTORY ────────────────
exports.getHistory = async (req, res) => {
  try {
    const result = await db.query(
      `SELECT * FROM generations
       WHERE user_id=$1
       ORDER BY created_at DESC`,
      [req.user.id]
    );
    res.json(result.rows);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
};

// ── CREATE GENERATION (Start AI job) ──────
exports.createGeneration = async (req, res) => {
  const { input_text } = req.body;
  if (!input_text) return res.status(400).json({ error: 'input_text is required.' });

  try {
    const result = await db.query(
      `INSERT INTO generations (user_id, input_text, status)
       VALUES ($1, $2, 'pending')
       RETURNING *`,
      [req.user.id, input_text]
    );
    const generation = result.rows[0];

    // TODO: trigger your AI model here (e.g. call Python service)
    // For now we simulate a notification
    await db.query(
      `INSERT INTO notifications (user_id, title, message, type)
       VALUES ($1, 'Video Ready', $2, 'success')`,
      [req.user.id, `Your sign language video for "${input_text}" is ready to view.`]
    );

    res.status(201).json(generation);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
};
