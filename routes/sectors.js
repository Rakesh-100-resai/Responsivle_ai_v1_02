const express = require('express');
const path = require('path');
const fs = require('fs').promises;
const router = express.Router();

// ✅ Ensure your sector data is defined or imported here
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

router.get('/', (req, res) => res.json(sectors));

router.get('/:id', (req, res) => {
  const sector = sectors.find(s => s.id === req.params.id);
  res.json(sector || { error: 'Sector not found' });
});

// ✅ RESTORED: application lookup route (was missing)
router.get('/application/:appId', (req, res) => {
  const app = sectors.flatMap(s => s.applications).find(a => a.id === req.params.appId);
  res.json(app || { error: 'Application not found' });
});

router.get('/datasets', async (req, res) => {
  const sectorId = req.query.sector;
  try {
    const metadata = JSON.parse(await fs.readFile(path.join(__dirname, '../datasets', 'metadata.json'), 'utf8'));
    const datasets = metadata.datasets.filter(ds => ds.sector === sectorId);
    res.json(datasets);
  } catch (err) {
    res.status(500).json({ error: 'Failed to load dataset metadata' });
  }
});

module.exports = router;
