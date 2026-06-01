const db = require('../config/db');

// GET ALL SIGNS (avec filtre catégorie)
exports.getSigns = async (req, res) => {
  const { category, search } = req.query;
  try {
    let query  = 'SELECT * FROM signs WHERE 1=1';
    const params = [];

    if (category) {
      params.push(category);
      query += ` AND category = $${params.length}`;
    }
    if (search) {
      params.push(`%${search}%`);
      query += ` AND word ILIKE $${params.length}`;
    }
    query += ' ORDER BY word ASC';

    const result = await db.query(query, params);
    res.json(result.rows);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
};

// GET ONE SIGN
exports.getSignById = async (req, res) => {
  try {
    const result = await db.query('SELECT * FROM signs WHERE id=$1', [req.params.id]);
    if (result.rows.length === 0)
      return res.status(404).json({ error: 'Sign not found.' });
    res.json(result.rows[0]);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
};

// ADD SIGN (admin usage)
exports.createSign = async (req, res) => {
  const { word, category, image_url, video_url } = req.body;
  if (!word || !category)
    return res.status(400).json({ error: 'word and category are required.' });

  try {
    const result = await db.query(
      'INSERT INTO signs (word, category, image_url, video_url) VALUES ($1,$2,$3,$4) RETURNING *',
      [word, category, image_url || null, video_url || null]
    );
    res.status(201).json(result.rows[0]);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
};
