# Sprint contact research

Status: research infrastructure only. No real labeled dataset has been ingested and no sprint-trained model has been released. The synthetic checks only exercise code paths; they are not accuracy measurements.

- `contact_model.js`: shared 60-value temporal feature extractor, multiclass classifier, event suppression and one-to-one event matching.
- `train_contact.js`: Node training CLI; athlete-disjoint train/validation/test, threshold selected on validation, held-out precision/recall/F1 and matched-event MAE/RMSE. Uses a 20 ms event matching tolerance; reports coverage, including events missed because features are unavailable. Matched timing error alone is insufficient.
- `../sprint_research.html`: optional browser research copy. Local draft export and optional experimental model loading. Existing measurements unchanged. No model bundled and no private footage uploaded.

## Dataset contract

JSON object with `clips`, each containing a unique pseudonymous `id`, `athlete`, `split` (train/validation/test), unique source reference (including clip boundaries), verified license, `groundTruth` (force or independent-video), `timeVerified: true`, `completeAnnotation: true`, width, height, playbackTimeScale, rows (seekSeconds, duplicate, points; MediaPipe 33 normalized landmarks with visibility), and events (side Left/Right, type touchdown/toeoff, t in real seconds from the same origin). App exports are unverified drafts; do not relabel as independent ground truth without independent review. Track provenance, athlete identities pseudonymously, trial/camera duplication, force/video synchronization and domain. Never publish personal footage or landmarks without consent.

Run with Node: `node research/train_contact.js dataset.json model.json evaluation.json`. Keep private data and model files local until licensing and disclosure are approved. Do not commit user footage or individual identifiers. 

## Next steps / blockers

- Audit licenses and obtain usable labels and matching 2D landmarks (FootNet, VideoRun2D, Bath biomechanics datasets; access, modality and reuse not yet verified in this implementation). No automatic assumption of usable outdoor sprint videos. Aim for 1,000 unique labeled clips if feasible, not frames.
- Train on available compliant data and compare the current rules against predictions on the same held-out athletes; separate treadmill and outdoor sprint cohorts, camera types, frame rates and start/upright phases. Evaluate false positives, missing events, timing uncertainty and contact interval accuracy, not only matched events.
- Add independent phase/stride model only after reliable data and calibration; metric speed/stride require spatial reference, even with learned detection.
- Release gate: held-out outdoor accuracy improvement with adequate coverage and no unacceptable regression, verified app functionality, no automatic deployment of a trained model. Existing app remains intact.

## Smoke tests recorded

- Feature extractor: 160 synthetic windows from 90 artificial samples, zero windows with all landmarks missing.
- Classifier: 90/90 artificial separable feature vectors classified on the same training set; not held-out real accuracy.
- Event matching: two predictions for one reference event count as one true positive and one false positive.
- Training CLI logic: three artificial clips exercised all splits; athlete leakage and unverified timing rejected; release flag remains false. No claimed synthetic accuracy or real training progress.
