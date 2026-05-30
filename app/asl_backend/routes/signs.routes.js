const router = require('express').Router();
const ctrl   = require('../controllers/signs.controller');
const auth   = require('../middleware/auth.middleware');

router.get('/',     auth, ctrl.getSigns);
router.get('/:id',  auth, ctrl.getSignById);
router.post('/',    auth, ctrl.createSign);

router.post('/generate', ctrl.generateSign);
module.exports = router;
