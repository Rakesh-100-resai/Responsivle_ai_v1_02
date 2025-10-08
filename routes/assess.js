const express = require('express');
const multer = require('multer');
const path = require('path');
const fs = require('fs').promises;
const { spawn } = require('child_process');
const { v4: uuid } = require('uuid');
const db = require('../utils/db');

const router = express.Router();
const upload = multer({ dest: 'uploads/' });

router.post(
  '/',
  upload.fields([
    { name: 'dataset', maxCount: 1 },
    { name: 'model', maxCount: 1 },
    { name: 'description', maxCount: 1 }
  ]),
  async (req, res) => {
    const { principles, modelType, dataset_id } = req.body;
    const datasetFile = req.files?.dataset?.[0];
    const modelFile = req.files?.model?.[0];

    if (!modelFile) {
      return res.status(400).json({ error: 'Model file is required' });
    }

    if (!datasetFile && !dataset_id) {
      return res.status(400).json({ error: 'Dataset file or dataset_id is required' });
    }

    let parsedPrinciples;
    try {
      parsedPrinciples = JSON.parse(principles);
      if (!Array.isArray(parsedPrinciples) || !parsedPrinciples.every(p => typeof p === 'string')) {
        throw new Error('Principles must be an array of strings');
      }
    } catch (err) {
      return res.status(400).json({ error: 'Invalid principles format' });
    }

    const datasetPath = dataset_id
      ? path.join(__dirname, '../datasets', `${dataset_id}.csv`)
      : datasetFile.path;
    const modelPath = modelFile.path;

    const validModelTypes = ['logistic_regression', 'random_forest', 'xgboost'];
    let mappedModelType = validModelTypes.find(type =>
      modelFile.originalname.toLowerCase().includes(type)
    ) || 'xgboost';

    const reportId = uuid();

    try {
      await db.run(
        'INSERT INTO reports (id, progress, dataset_id) VALUES (?, 0, ?)',
        [reportId, dataset_id || 'custom']
      );

      const spawnArgs = [
        'evaluate_model.py',
        modelPath,
        datasetPath,
        JSON.stringify(parsedPrinciples)
      ];
      if (mappedModelType) {
        spawnArgs.push('--model_type', mappedModelType);
      }

      // const proc = spawn('python3', spawnArgs);
      const proc = spawn(
        'C:\\Users\\IFOCUS\\AppData\\Local\\Programs\\Python\\Python310\\python.exe',
        spawnArgs
      );



      let output = '';
      let errorOutput = '';

      proc.stdout.on('data', data => (output += data.toString()));
      proc.stderr.on('data', data => (errorOutput += data.toString()));

      const result = await new Promise((resolve, reject) => {
        proc.on('close', code => {
          if (code === 0) {
            try {
              const report = JSON.parse(output);
              resolve({ reportId, report, principles: parsedPrinciples });
            } catch (err) {
              reject(new Error('Invalid JSON from Python: ' + err.message));
            }
          } else {
            reject(new Error(`Python script failed (code ${code}): ${errorOutput}`));
          }
        });
      });

      await db.run(
        'UPDATE reports SET progress = 100, scores = ? WHERE id = ?',
        [JSON.stringify(result.report), reportId]
      );

      if (datasetFile?.path) await fs.unlink(datasetFile.path).catch(() => {});
      if (modelFile?.path) await fs.unlink(modelFile.path).catch(() => {});

      res.json(result);
    } catch (err) {
      console.error('Assessment Error:', err);
      res.status(500).json({ error: err.message });
    }
  }
);

router.get('/status/:reportId', async (req, res) => {
  try {
    const row = await db.get('SELECT progress FROM reports WHERE id = ?', req.params.reportId);
    res.json({ progress: row?.progress || 0 });
  } catch (err) {
    res.status(500).json({ error: 'Could not fetch report status' });
  }
});

router.get('/report/:reportId', async (req, res) => {
  try {
    const row = await db.get('SELECT scores FROM reports WHERE id = ?', req.params.reportId);
    if (row?.scores) {
      res.json({ scores: JSON.parse(row.scores) });
    } else {
      res.status(404).json({ error: 'Report not found' });
    }
  } catch (err) {
    res.status(500).json({ error: 'Could not fetch report data' });
  }
});

module.exports = router;
