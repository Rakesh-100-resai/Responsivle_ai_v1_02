// utils/db.js
const sqlite3 = require('sqlite3').verbose();
const path = require('path');

// Resolve the database path
const dbPath = path.join(__dirname, '../assessments.db');

// Initialize the database
const db = new sqlite3.Database(dbPath, (err) => {
  if (err) {
    console.error('Failed to connect to SQLite database:', err.message);
  } else {
    console.log('Connected to SQLite database at', dbPath);
  }
});

// Optional: Create the reports table if not exists (safe for re-use)
db.run(`
  CREATE TABLE IF NOT EXISTS reports (
    id TEXT PRIMARY KEY,
    sector_id TEXT,
    application_id TEXT,
    dataset_id TEXT,
    scores JSON,
    progress INTEGER,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
  )
`, (err) => {
  if (err) {
    console.error('Failed to initialize reports table:', err.message);
  } else {
    console.log('Reports table is ready.');
  }
});

module.exports = db;
