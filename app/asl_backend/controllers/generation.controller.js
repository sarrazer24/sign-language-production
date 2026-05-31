const db = require('../config/db');

// CREATE GENERATION (soumettre un texte)
exports.createGeneration = async (req, res) => {
  const { original_text } = req.body;
  if (!original_text)
    return res.status(400).json({ error: 'original_text is required.' });

  try {
    const result = await db.query(
      `INSERT INTO generations (user_id, original_text, status)
       VALUES ($1, $2, 'pending') RETURNING *`,
      [req.user.id, original_text]
    );
    const generation = result.rows[0];

    // Créer une notification automatique
    await db.query(
      `INSERT INTO notifications (user_id, type, title, message, generation_id)
       VALUES ($1, 'info', 'Processing...', $2, $3)`,
      [req.user.id, `Your video for "${original_text}" is being generated.`, generation.id]
    );

    res.status(201).json(generation);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
};

// GET ALL GENERATIONS (historique de l'user)
exports.getMyGenerations = async (req, res) => {
  try {
    const result = await db.query(
      'SELECT * FROM generations WHERE user_id=$1 ORDER BY created_at DESC',
      [req.user.id]
    );
    res.json(result.rows);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
};

// GET ONE GENERATION
exports.getGenerationById = async (req, res) => {
  try {
    const result = await db.query(
      'SELECT * FROM generations WHERE id=$1 AND user_id=$2',
      [req.params.id, req.user.id]
    );
    if (result.rows.length === 0)
      return res.status(404).json({ error: 'Generation not found.' });
    res.json(result.rows[0]);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
};

// UPDATE STATUS (appelé par le service IA)
exports.updateStatus = async (req, res) => {
  const { status, video_url } = req.body;
  const allowed = ['pending', 'processing', 'ready', 'failed'];
  if (!allowed.includes(status))
    return res.status(400).json({ error: 'Invalid status.' });

  try {
    const result = await db.query(
      'UPDATE generations SET status=$1, video_url=$2 WHERE id=$3 RETURNING *',
      [status, video_url || null, req.params.id]
    );
    if (result.rows.length === 0)
      return res.status(404).json({ error: 'Generation not found.' });

    const gen = result.rows[0];

    // Notification automatique selon le statut
    if (status === 'ready') {
      await db.query(
        `INSERT INTO notifications (user_id, type, title, message, generation_id)
         VALUES ($1, 'video_ready', 'Video Ready', $2, $3)`,
        [gen.user_id, `Your sign language video for "${gen.original_text}" is ready to view.`, gen.id]
      );
    } else if (status === 'failed') {
      await db.query(
        `INSERT INTO notifications (user_id, type, title, message, generation_id)
         VALUES ($1, 'generation_failed', 'Generation Failed', $2, $3)`,
        [gen.user_id, `We couldn't process your request for "${gen.original_text}". Please try again.`, gen.id]
      );
    }

    res.json(gen);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
};

// RATE A VIDEO
exports.rateGeneration = async (req, res) => {
  const { score } = req.body;
  if (!score || score < 1 || score > 5)
    return res.status(400).json({ error: 'Score must be between 1 and 5.' });

  try {
    const result = await db.query(
      `INSERT INTO ratings (user_id, generation_id, score)
       VALUES ($1, $2, $3)
       ON CONFLICT (user_id, generation_id) DO UPDATE SET score=$3
       RETURNING *`,
      [req.user.id, req.params.id, score]
    );
    res.json(result.rows[0]);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
};
exports.analyzeVideo = async (req, res) => {
  try {
    // La vidéo sera traitée par le modèle IA de ta camarade
    // Pour l'instant on sauvegarde juste la génération
    const result = await db.query(
      `INSERT INTO generations (user_id, original_text, status)
       VALUES ($1, $2, 'processing') RETURNING *`,
      [req.user.id, 'Video analysis']
    );
    res.json({ 
      message: 'Video received',
      generation: result.rows[0],
      text: '' // sera rempli par le modèle IA
    });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
};