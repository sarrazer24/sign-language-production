const router = require('express').Router();
const ctrl   = require('../controllers/generation.controller');
const auth   = require('../middleware/auth.middleware');

router.post('/',                    auth, ctrl.createGeneration);
router.get('/',                     auth, ctrl.getMyGenerations);
router.get('/:id',                  auth, ctrl.getGenerationById);
router.patch('/:id/status',         auth, ctrl.updateStatus);
router.post('/:id/rate',            auth, ctrl.rateGeneration);
router.post('/analyze-video', auth, ctrl.analyzeVideo);

module.exports = router;
