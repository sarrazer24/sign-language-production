const db = require('../config/db');

// GET RECENT ACTIVITY (home screen)
exports.getRecentActivity = async (req, res) => {
  try {
    const result = await db.query(
      `SELECT g.id, g.original_text, g.status, g.video_url, g.created_at
       FROM generations g
       WHERE g.user_id = $1
       ORDER BY g.created_at DESC
       LIMIT 10`,
      [req.user.id]
    );
    res.json(result.rows);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
};
