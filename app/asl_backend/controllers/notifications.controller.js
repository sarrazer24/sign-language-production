const db = require('../config/db');

// GET ALL NOTIFICATIONS
exports.getNotifications = async (req, res) => {
  try {
    const result = await db.query(
      'SELECT * FROM notifications WHERE user_id=$1 ORDER BY created_at DESC',
      [req.user.id]
    );
    res.json(result.rows);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
};

// MARK AS READ
exports.markAsRead = async (req, res) => {
  try {
    await db.query(
      'UPDATE notifications SET is_read=TRUE WHERE id=$1 AND user_id=$2',
      [req.params.id, req.user.id]
    );
    res.json({ success: true });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
};

// MARK ALL AS READ
exports.markAllAsRead = async (req, res) => {
  try {
    await db.query(
      'UPDATE notifications SET is_read=TRUE WHERE user_id=$1',
      [req.user.id]
    );
    res.json({ success: true });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
};

// DELETE ALL NOTIFICATIONS
exports.clearAll = async (req, res) => {
  try {
    await db.query('DELETE FROM notifications WHERE user_id=$1', [req.user.id]);
    res.json({ success: true });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
};
