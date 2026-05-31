const bcrypt = require('bcryptjs');
const jwt    = require('jsonwebtoken');
const db     = require('../config/db');
const nodemailer = require('nodemailer');

// ── Email transporter ──
const transporter = nodemailer.createTransport({
  service: 'gmail',
  auth: {
    user: process.env.EMAIL_USER,
    pass: process.env.EMAIL_PASS,
  },
});

// SIGN UP
exports.signUp = async (req, res) => {
  const { full_name, email, password } = req.body;
  if (!full_name || !email || !password)
    return res.status(400).json({ error: 'All fields are required.' });

  try {
    const exists = await db.query('SELECT id FROM users WHERE email=$1', [email]);
    if (exists.rows.length > 0)
      return res.status(409).json({ error: 'Email already in use.' });

    const hashed = await bcrypt.hash(password, 10);
    const result = await db.query(
      'INSERT INTO users (full_name, email, password) VALUES ($1,$2,$3) RETURNING id, full_name, email',
      [full_name, email, hashed]
    );
    const user  = result.rows[0];
    const token = jwt.sign({ id: user.id, email: user.email }, process.env.JWT_SECRET, {
      expiresIn: process.env.JWT_EXPIRES_IN,
    });
    res.status(201).json({ token, user });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
};

// SIGN IN
exports.signIn = async (req, res) => {
  const { email, password } = req.body;
  if (!email || !password)
    return res.status(400).json({ error: 'Email and password required.' });

  try {
    const result = await db.query('SELECT * FROM users WHERE email=$1', [email]);
    if (result.rows.length === 0)
      return res.status(404).json({ error: 'User not found.' });

    const user  = result.rows[0];
    const match = await bcrypt.compare(password, user.password);
    if (!match) return res.status(401).json({ error: 'Wrong password.' });

    const token = jwt.sign({ id: user.id, email: user.email }, process.env.JWT_SECRET, {
      expiresIn: process.env.JWT_EXPIRES_IN,
    });
    res.json({ token, user: { id: user.id, full_name: user.full_name, email: user.email } });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
};

// GET PROFILE
exports.getProfile = async (req, res) => {
  try {
    const result = await db.query(
      'SELECT id, full_name, email, created_at FROM users WHERE id=$1',
      [req.user.id]
    );
    if (result.rows.length === 0)
      return res.status(404).json({ error: 'User not found.' });
    res.json(result.rows[0]);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
};

// FORGOT PASSWORD
exports.forgotPassword = async (req, res) => {
  const { email } = req.body;
  if (!email) return res.status(400).json({ error: 'Email required.' });

  try {
    const result = await db.query('SELECT id, full_name FROM users WHERE email=$1', [email]);
    if (result.rows.length === 0)
      return res.status(404).json({ error: 'Email not found.' });

    const user = result.rows[0];
    const resetToken = jwt.sign({ id: user.id, email }, process.env.JWT_SECRET, { expiresIn: '1h' });

    await transporter.sendMail({
      from: `"ASL App" <${process.env.EMAIL_USER}>`,
      to: email,
      subject: 'Reset your password',
      html: `
        <div style="font-family:Arial,sans-serif;max-width:500px;margin:auto;padding:32px;background:#f8f7ff;border-radius:16px;">
          <h2 style="color:#5B4FCF;">Reset your password</h2>
          <p>Hi ${user.full_name},</p>
          <p>We received a request to reset your password.</p>
          <p>Your reset token (valid 1 hour):</p>
          <div style="background:#fff;padding:16px;border-radius:8px;font-size:18px;font-weight:bold;color:#5B4FCF;text-align:center;letter-spacing:4px;">
            ${resetToken.slice(-8).toUpperCase()}
          </div>
          <p style="color:grey;font-size:12px;margin-top:24px;">If you didn't request this, ignore this email.</p>
        </div>
      `,
    });

    res.json({ message: 'Reset link sent.' });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
};
exports.resetPassword = async (req, res) => {
  const { email, new_password } = req.body;
  if (!email || !new_password)
    return res.status(400).json({ error: 'Email and new password required.' });

  try {
    const hashed = await bcrypt.hash(new_password, 10);
    await db.query('UPDATE users SET password=$1 WHERE email=$2', [hashed, email]);
    res.json({ message: 'Password reset successfully.' });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
};