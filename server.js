const express = require('express');
const multer = require('multer');
const sqlite3 = require('sqlite3').verbose();
const { spawn } = require('child_process');
const { v4: uuid } = require('uuid');
const path = require('path');
const fs = require('fs').promises;

const app = express();
const upload = multer({ dest: 'uploads/' });
const db = new sqlite3.Database('assessments.db');
port = 5000;

// Initialize database
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
`);

// Middleware
app.use(express.static(path.join(__dirname, 'public')));
app.use(express.json());

// Serve login page at root
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'login.html'));
});

// Sector and application data
const sectors = [
  { id: 'healthcare', title: 'Healthcare', icon: 'healthcare.svg', summary: 'AI influences diagnosis and treatment.', description: 'AI transforms healthcare with diagnostics.', applications: [
    { id: 'diagnostics', name: 'AI-assisted Diagnostics', description: 'AI for radiology and pathology.', challenges: 'Bias across demographics.', principles: ['Fairness', 'Transparency', 'Privacy'] }
  ]},
  { id: 'finance', title: 'Finance & Banking', icon: 'finance.svg', summary: 'AI decides loan approvals.', description: 'AI drives financial decisions.', applications: [
    { id: 'loan_approval', name: 'Loan Approval System', description: 'Automated loan approvals.', challenges: 'Non-discrimination.', principles: ['Fairness', 'Auditability'] }
  ]},
  { id: 'hr', title: 'Recruitment & HR', icon: 'hr.svg', summary: 'AI influences hiring.', description: 'AI shapes recruitment processes.', applications: [
    { id: 'resume_screening', name: 'Resume Screening', description: 'Automated resume analysis.', challenges: 'Gender/ethnic bias.', principles: ['Fairness', 'Transparency'] }
  ]},
  { id: 'law', title: 'Law Enforcement', icon: 'law.svg', summary: 'AI predicts crime risks.', description: 'AI in public safety.', applications: [
    { id: 'crime_prediction', name: 'Crime Prediction', description: 'Predictive policing.', challenges: 'Racial profiling.', principles: ['Fairness', 'Explainability'] }
  ]},
  { id: 'education', title: 'Education', icon: 'education.svg', summary: 'AI personalizes learning.', description: 'AI enhances education.', applications: [
    { id: 'adaptive_learning', name: 'Adaptive Learning', description: 'Personalized learning platforms.', challenges: 'Equal access.', principles: ['Fairness', 'Privacy'] }
  ]}
];

// API Endpoints
app.get('/api/sectors', (req, res) => res.json(sectors));


app.get('/api/sectors/:id', (req, res) => {
  const sector = sectors.find(s => s.id === req.params.id);
  res.json(sector || { error: 'Sector not found' });
});

app.get('/api/applications/:id', (req, res) => {
  const app = sectors.flatMap(s => s.applications).find(a => a.id === req.params.id);
  res.json(app || { error: 'Application not found' });
});

app.get('/api/datasets', async (req, res) => {
  const sectorId = req.query.sector;
  try {
    const metadata = JSON.parse(await fs.readFile(path.join(__dirname, 'datasets', 'metadata.json'), 'utf8'));
    const datasets = metadata.datasets.filter(ds => ds.sector === sectorId);
    res.json(datasets);
  } catch (err) {
    res.status(500).json({ error: 'Failed to load dataset metadata' });
  }
});

app.post('/api/login', (req, res) => {
  if (req.body.username === 'admin' && req.body.password === 'password') {
    res.json({ success: true });
  } else {
    res.status(401).json({ error: 'Invalid credentials' });
  }
});

app.post('/api/assess', upload.fields([
  { name: 'dataset', maxCount: 1 },
  { name: 'model', maxCount: 1 },
  { name: 'description', maxCount: 1 }
]), async (req, res) => {
  const { principles, modelType, dataset_id } = req.body;
  const datasetFile = req.files?.dataset?.[0];
  const modelFile = req.files?.model?.[0];

  // Validate inputs
  if (!modelFile) {
    return res.status(400).json({ error: 'Model file is required' });
  }
  if (!datasetFile && !dataset_id) {
    return res.status(400).json({ error: 'Dataset file or dataset_id is required' });
  }

  // Validate principles
  let parsedPrinciples;
  try {
    parsedPrinciples = JSON.parse(principles);
    if (!Array.isArray(parsedPrinciples) || !parsedPrinciples.every(p => typeof p === 'string')) {
      throw new Error('Principles must be an array of strings');
    }
    console.log('Parsed principles:', parsedPrinciples);
  } catch (error) {
    console.error('Invalid principles format:', principles, error);
    return res.status(400).json({ error: 'Principles must be a valid JSON array of strings' });
  }

  const datasetPath = dataset_id ? path.join(__dirname, 'datasets', `${dataset_id}.csv`) : datasetFile.path;
  const modelPath = modelFile.path;

  // Map modelType to valid evaluation.py model types
  const validModelTypes = ['logistic_regression', 'random_forest', 'xgboost'];
  let mappedModelType = null;
  if (modelType === 'loan_approval') {
    mappedModelType = validModelTypes.find(type => modelFile.originalname.toLowerCase().includes(type)) || 'xgboost';
  } else if (validModelTypes.includes(modelType)) {
    mappedModelType = modelType;
  }

  const reportId = uuid();

  try {
    await db.run('INSERT INTO reports (id, progress, dataset_id) VALUES (?, 0, ?)', [reportId, dataset_id || 'custom']);

    // Construct spawn arguments
    const spawnArgs = [
      'evaluate_model.py',
      modelPath,
      datasetPath,
      JSON.stringify(parsedPrinciples) // Use parsed and re-stringified principles
    ];
    if (mappedModelType) {
      spawnArgs.push('--model_type', mappedModelType);
    }

    console.log('Spawn arguments:', spawnArgs); // Debug spawn args

    const proc = spawn('python3', spawnArgs);

    let output = '';
    let errorOutput = '';

    proc.stdout.on('data', data => {
      output += data;
      console.log('Python stdout:', data.toString());
    });
    proc.stderr.on('data', data => {
      errorOutput += data;
      console.error('Python stderr:', data.toString());
    });

    const result = await new Promise((resolve, reject) => {
      proc.on('close', code => {
        if (code === 0) {
          try {
            const evaluationOutput = JSON.parse(output);
            resolve({
              reportId,
              report: evaluationOutput,
              principles: parsedPrinciples
            });
          } catch (err) {
            reject(new Error(`Invalid JSON output from Python script: ${err.message}\nSTDERR: ${errorOutput}`));
          }
        } else {
          reject(new Error(`Evaluation failed with code ${code}: ${errorOutput}`));
        }
      });
    });

    await db.run('UPDATE reports SET progress = 100, scores = ? WHERE id = ?', [JSON.stringify(result.report), reportId]);
    
    // Clean up uploaded files
    if (datasetFile?.path) await fs.unlink(datasetFile.path).catch(() => {});
    if (modelFile?.path) await fs.unlink(modelFile.path).catch(() => {});

    res.json(result);
  } catch (error) {
    console.error('Assessment error:', error);
    res.status(400).json({ error: error.message });
  }
});

app.get('/api/assess/status/:reportId', async (req, res) => {
  const report = await new Promise(resolve => db.get('SELECT progress FROM reports WHERE id = ?', [req.params.reportId], (err, row) => resolve(row)));
  res.json({ progress: report ? report.progress : 0 });
});
app.get('/api/reports/:reportId', async (req, res) => {
  const report = await new Promise(resolve => db.get('SELECT scores FROM reports WHERE id = ?', [req.params.reportId], (err, row) => resolve(row)));
  res.json(report ? { scores: JSON.parse(report.scores) } : { error: 'Report not found' });
});

// Error handling for unmatched routes
app.use((req, res) => {
  res.status(404).json({ error: `Cannot ${req.method} ${req.url}` });
});

//app.listen(5000, () => console.log('Server running on port 5000'));
app.listen(port, () => {
  console.log(`Server running at http://localhost:${port}`);
});