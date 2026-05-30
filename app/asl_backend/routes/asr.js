const express  = require('express');
const multer   = require('multer');
const FormData = require('form-data');
const axios    = require('axios');
const fs       = require('fs');

const router  = express.Router();
const upload  = multer({ dest: 'uploads/' });
const ASR_URL = process.env.ASR_SERVICE_URL || 'http://localhost:9000';
router.post('/transcribe', upload.single('audio'), async (req, res) => {
  if (!req.file) return res.status(400).json({ error: 'No audio file' });

  const form = new FormData();
  form.append('file', fs.createReadStream(req.file.path), req.file.originalname);

  try {
    const { data } = await axios.post(
      `${ASR_URL}/transcribe`,
      form,
      { headers: form.getHeaders() }
    );
    res.json({ transcript: data.transcript });   // ← send back to Flutter
  } catch (err) {
    res.status(500).json({ error: err.message });
  } finally {
    fs.unlinkSync(req.file.path);   // always clean up
  }
});
module.exports = router;