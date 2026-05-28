const router = require('express').Router();
const ctrl   = require('../controllers/activity.controller');
const auth   = require('../middleware/auth.middleware');

router.get('/', auth, ctrl.getRecentActivity);

module.exports = router;
