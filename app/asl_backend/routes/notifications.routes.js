const router = require('express').Router();
const ctrl   = require('../controllers/notifications.controller');
const auth   = require('../middleware/auth.middleware');

router.get('/',              auth, ctrl.getNotifications);
router.patch('/:id/read',    auth, ctrl.markAsRead);
router.patch('/read-all',    auth, ctrl.markAllAsRead);
router.delete('/clear',      auth, ctrl.clearAll);

module.exports = router;
