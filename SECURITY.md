# Security and Privacy

## Supported Version

Security fixes are applied to the latest commit on **main**.

## Reporting

Do not publish vulnerabilities or exposed medical data in a public issue.
Contact the repository owner privately through their GitHub profile and include
minimal reproduction details.

## Medical Data

Prescription images may contain names, contact details, diagnoses, signatures,
and medication history.

- Use de-identified research data.
- Do not commit user uploads or Gradio temporary files.
- Do not send images to external services without explicit consent.
- Remove metadata and patient identifiers before sharing.
- Apply access controls and retention limits in deployments.

## Model Safety

OCR and GAN outputs are untrusted text. Never use them to automatically issue,
change, or validate a prescription. Display raw results and require qualified
human review.
