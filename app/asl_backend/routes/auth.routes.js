const router  = require('express').Router();
const ctrl    = require('../controllers/auth.controller');
const auth    = require('../middleware/auth.middleware');

router.post('/signup',  ctrl.signUp);
router.post('/signin',  ctrl.signIn);
router.get('/profile',  auth, ctrl.getProfile);
router.post('/forgot-password', ctrl.forgotPassword);
router.post('/reset-password', ctrl.resetPassword);

module.exports = router;
