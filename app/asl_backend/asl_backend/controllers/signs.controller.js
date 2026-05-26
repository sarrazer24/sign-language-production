const db = require('../config/db');

// ── GET ALL SIGNS (with optional category filter) ──
exports.getSigns = async (req, res) => {
  const { category } = req.query; // alphabet | numbers | phrases
  try {
    let query  = 'SELECT * FROM signs';
    let params = [];
    if (category) {
      query  += ' WHERE category=$1';
      params  = [category];
    }
    query += ' ORDER BY letter ASC, word ASC';
    const result = await db.query(query, params);
    res.json(result.rows);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
};

// ── SEARCH SIGNS ──────────────────────────
exports.searchSigns = async (req, res) => {
  const { q } = req.query;
  try {
    const result = await db.query(
      `SELECT * FROM signs
       WHERE letter ILIKE $1 OR word ILIKE $1
       ORDER BY letter ASC`,
      [`%${q}%`]
    );
    res.json(result.rows);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
};

// ── GET SINGLE SIGN ───────────────────────
exports.getSignById = async (req, res) => {
  try {
    const result = await db.query('SELECT * FROM signs WHERE id=$1', [req.params.id]);
    if (result.rows.length === 0) return res.status(404).json({ error: 'Sign not found.' });
    res.json(result.rows[0]);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
};
