const router = require('express').Router();
const auth   = require('../middleware/auth.middleware');
const ctrl   = require('../controllers/signs.controller');

router.get('/',        auth, ctrl.getSigns);
router.get('/search',  auth, ctrl.searchSigns);
router.get('/:id',     auth, ctrl.getSignById);

module.exports = router;
