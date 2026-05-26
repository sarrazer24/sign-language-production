const router = require('express').Router();
const auth   = require('../middleware/auth.middleware');
const ctrl   = require('../controllers/activity.controller');

router.get('/notifications',          auth, ctrl.getNotifications);
router.patch('/notifications/:id',    auth, ctrl.markAsRead);
router.delete('/notifications',       auth, ctrl.clearAll);

router.get('/history',                auth, ctrl.getHistory);
router.post('/generate',              auth, ctrl.createGeneration);

module.exports = router;
