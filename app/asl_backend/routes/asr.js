const express  = require('express');
const multer   = require('multer');
const FormData = require('form-data');
const axios    = require('axios');
const fs       = require('fs');

const router = express.Router();
const upload = multer({ dest: 'uploads/' });

const ASR_URL =
  process.env.ASR_SERVICE_URL ||
  'https://muster-gladiator-tracing.ngrok-free.dev';

router.post('/', upload.single('audio'), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: 'No audio file' });
  }

  const form = new FormData();
  form.append(
    'file',
    fs.createReadStream(req.file.path),
    req.file.originalname
  );

  try {
    const { data } = await axios.post(
      `${ASR_URL}/transcribe`,
      form,
      {
        headers: {
          ...form.getHeaders(),
          'ngrok-skip-browser-warning': 'true',
        },
        timeout: 60000,
      }
    );

    res.json({ transcript: data.transcript });
  } catch (err) {
    res.status(500).json({
      error: err.message,
    });
  } finally {
    if (fs.existsSync(req.file.path)) {
      fs.unlinkSync(req.file.path);
    }
  }
});

module.exports = router;