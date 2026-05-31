require('dotenv').config();
const express = require('express');
const cors    = require('cors');

const authRoutes       = require('./routes/auth.routes');
const signsRoutes      = require('./routes/signs.routes');
const activityRoutes   = require('./routes/activity.routes');
const notifRoutes      = require('./routes/notifications.routes');
const generationRoutes = require('./routes/generation.routes');

const app = express();

app.use(cors({
  origin: '*',
  methods: ['GET', 'POST', 'PUT', 'PATCH', 'DELETE'],
  allowedHeaders: ['Content-Type', 'Authorization'],
}));
app.use(express.json());

app.use('/api/auth',          authRoutes);
app.use('/api/signs',         signsRoutes);
app.use('/api/activity',      activityRoutes);
app.use('/api/notifications', notifRoutes);
app.use('/api/generations',   generationRoutes);

app.get('/', (req, res) => res.json({ status: 'ASL API running ✅' }));

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));
