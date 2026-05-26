const router = require('express').Router();
const auth   = require('../middleware/auth.middleware');
const ctrl   = require('../controllers/auth.controller');

router.post('/signup',          ctrl.signUp);
router.post('/signin',          ctrl.signIn);
router.post('/forgot-password', ctrl.forgotPassword);
router.get('/profile',  auth,   ctrl.getProfile);

module.exports = router;
