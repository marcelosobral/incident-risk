# Tricura Incident Risk

## Overview
This project builds a resident-level risk model to predict whether a resident will experience any incident in the next 30 days. The goal is to help skilled nursing facilities identify high-risk residents early and apply preventive interventions that reduce incident frequency and severity.

## Discovery
- Data is spread across 17 parquet tables with a mix of high-volume longitudinal signals (vitals, medications, functional responses) and smaller event tables (incidents, injuries).
- The incidents table has 3,578 records across 987 residents and 92 facilities.
- Incident types are highly imbalanced: Falls account for about 73.5% of incidents, followed by Wounds (~17.5%) and Altercations (~7.2%).
- Incident volume ramps up in mid-2023 and stays fairly stable through early 2025.

## Methodology
- Frame the task as binary classification with a 30-day prediction horizon.
- Build a resident-day (or resident-week) timeline and generate features using only data available prior to the prediction date to avoid leakage.
- Use a time-based train/validation split (train on earlier periods, validate on later periods).

## Modeling Plan
- Start with a baseline model (logistic regression) and progress to LightGBM for stronger tabular performance.
- Prioritize interpretability and calibrated risk scores to enable actionability.
- Track performance using AUROC and PR-AUC, and include an operational metric such as Recall at Top 10% risk.

## Validation
- Use temporal validation (no random splits).
- Monitor stability of incident rates across splits and check calibration in the validation window.

## Business Application
- Use the risk score to prioritize residents for targeted interventions (e.g., fall prevention plans, medication reviews, mobility support).
- Estimate potential savings by comparing incident reduction in the top-risk cohort against average incident costs.

## Brief EDA Highlights
- Largest tables: vitals (2.5M rows), medications (1.4M), gg_responses (660k), document_tags (562k), adl_responses (480k).
- Time coverage is broad: incidents span 2019-2025, while several care/clinical tables extend into 2026.
- Missingness is concentrated in expected columns (e.g., resolved_at, deceased_date, optional fields).
- Data quality notes: timestamp outliers exist (e.g., physician_orders.start_at in 1855) and location categories show duplicates/near-duplicates, so normalization and date correction rules are required.

## Next Steps
- Formalize the resident timeline dataset with strict pre-incident feature windows.
- Implement heuristic timestamp correction (e.g., year shifts or month/day swaps) for flagged outliers while retaining raw fields for traceability.
- Build feature groups across incidents, medications, diagnoses, functional status, vitals, and care plans.
- Train, validate, and interpret the model, then summarize results and business impact.

## Appendix

### Appendix A: Data Dictionary
Descriptions and notes are inferred from column names; example values come from a small sample of the data. Nullable reflects the Parquet schema (whether nulls are allowed).

### adl_responses
| Table name | Column name | Data type | Description | Example value | Nullable | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| adl_responses | adl_response_id | string | Identifier for adl response | ba6980c7-9fa3-50a0-9675-0e04f20c1d87 | yes | Join key |
| adl_responses | resident_id | string | Identifier for resident | 3fe20e80-80b2-50ea-8d18-966b6c016b05 | yes | Join key |
| adl_responses | facility_id | string | Identifier for facility | 3fe507e8-7058-5b34-b9c4-8332df85c9dc | yes | Join key |
| adl_responses | assessment_date | timestamp | Timestamp for assessment date | 2025-01-31 00:00:00 | yes |  |
| adl_responses | activity | string | Activity | Eating - Self-Performance | yes |  |
| adl_responses | category | string | Category | Self-Performance | yes |  |
| adl_responses | response | string | Response | 0 | yes |  |
| adl_responses | response_description | string | Response description | INDEPENDENT - No help or staff oversight at any time | yes |  |
| adl_responses | response_status | string | Response status | Complete | yes |  |
| adl_responses | previous_response | string | Previous response | 0 | yes |  |
| adl_responses | adl_change | int | Adl change | 0 | yes |  |
| adl_responses | created_at | timestamp | Timestamp for created at | 2025-01-31 21:27:00 | yes | System timestamp |

### care_plans
| Table name | Column name | Data type | Description | Example value | Nullable | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| care_plans | care_plan_id | string | Identifier for care plan | 992389a1-ca64-5065-92e8-901fed3938d3 | yes | Join key |
| care_plans | resident_id | string | Identifier for resident | 333ee797-476f-5d73-8a20-02c109f8abb5 | yes | Join key |
| care_plans | facility_id | string | Identifier for facility | 018d5b79-b86a-5dc8-8453-417d99a8f1f9 | yes | Join key |
| care_plans | initiated_at | timestamp | Timestamp for initiated at | 2025-01-10 01:00:00 | yes |  |
| care_plans | closed_reason | string | Closed reason | Discharge | yes |  |
| care_plans | closed_at | timestamp | Timestamp for closed at | 2025-07-10 14:00:00 | yes |  |
| care_plans | next_review_at | timestamp | Timestamp for next review at | 2025-12-01 23:00:00 | yes |  |
| care_plans | strikeout | boolean | Strikeout | False | yes |  |
| care_plans | created_at | timestamp | Timestamp for created at | 2025-01-10 10:25:09.130000 | yes | System timestamp |

### diagnoses
| Table name | Column name | Data type | Description | Example value | Nullable | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| diagnoses | diagnosis_id | string | Identifier for diagnosis | 04a797cb-07ac-59ea-81fa-e79f896c5404 | yes | Join key |
| diagnoses | facility_id | string | Identifier for facility | 018d5b79-b86a-5dc8-8453-417d99a8f1f9 | yes | Join key |
| diagnoses | resident_id | string | Identifier for resident | fa18f828-c5df-528f-a8b8-1263248df1e2 | yes | Join key |
| diagnoses | icd_10_code | string | Icd 10 code | R05.1 | yes |  |
| diagnoses | onset_at | timestamp | Timestamp for onset at | 2025-12-22 00:00:00 | yes |  |
| diagnoses | resolved_at | timestamp | Timestamp for resolved at |  | yes |  |
| diagnoses | strikeout | boolean | Strikeout | False | yes |  |
| diagnoses | created_at | timestamp | Timestamp for created at | 2025-12-22 17:37:16.587000 | yes | System timestamp |

### document_tags
| Table name | Column name | Data type | Description | Example value | Nullable | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| document_tags | document_tag_id | string | Identifier for document tag | 6a7f4f03-2173-4041-9808-c80b5db6339a | yes | Join key |
| document_tags | resident_id | string | Identifier for resident | 072691ff-60ba-5d4c-a8b4-ba582a99d803 | yes | Join key |
| document_tags | facility_id | string | Identifier for facility | 080d2f83-a504-5500-a4ad-1efa311b6fef | yes | Join key |
| document_tags | doc_type | string | Doc type | physician_orders | yes |  |
| document_tags | tag_id | string | Identifier for tag | o2_sat_order | yes | Join key |
| document_tags | match_confidence | float | Match confidence | 0.9990600943565369 | yes |  |
| document_tags | editable | boolean | Editable | True | yes |  |
| document_tags | created_at | timestamp | Timestamp for created at | 2023-08-29 10:17:16.598654 | yes | System timestamp |
| document_tags | deleted_at | timestamp | Timestamp for deleted at |  | yes | System timestamp |

### factors
| Table name | Column name | Data type | Description | Example value | Nullable | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| factors | factor_id | string | Identifier for factor | 29cede3a-3d34-56f2-ae25-753f55600f8a | yes | Join key |
| factors | incident_id | string | Identifier for incident | 7036c975-5086-5a59-a092-318f3a1f26b4 | yes | Join key |
| factors | facility_id | string | Identifier for facility | 0240d706-3348-5117-8d03-b06c5141e8c0 | yes | Join key |
| factors | factor_type | string | Factor type | Predisposing Situation | yes |  |
| factors | created_at | timestamp | Timestamp for created at | 2024-10-31 15:04:31.587000 | yes | System timestamp |

### gg_responses
| Table name | Column name | Data type | Description | Example value | Nullable | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| gg_responses | gg_response_id | string | Identifier for gg response | fe57c7f3-3d7d-5e42-85db-689da57ef54a | yes | Join key |
| gg_responses | facility_id | string | Identifier for facility | 080d2f83-a504-5500-a4ad-1efa311b6fef | yes | Join key |
| gg_responses | resident_id | string | Identifier for resident | d9631253-270e-50ff-9c04-1fe9aa4f8d9a | yes | Join key |
| gg_responses | task_group | string | Task group | Mobility | yes |  |
| gg_responses | task_name | string | Task name | Sit to Stand | yes |  |
| gg_responses | response | string | Response | Not Applicable | yes |  |
| gg_responses | response_code | int | Response code | 9 | yes |  |
| gg_responses | previous_response_code | int | Previous response code | 9 | yes |  |
| gg_responses | change | int | Change | 0.0 | yes |  |
| gg_responses | created_at | timestamp | Timestamp for created at | 2025-01-31 15:16:00 | yes | System timestamp |

### hospital_admissions
| Table name | Column name | Data type | Description | Example value | Nullable | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| hospital_admissions | admission_id | string | Identifier for admission | 534a6599-4203-5262-97fa-e6a2662d4e80 | yes | Join key |
| hospital_admissions | facility_id | string | Identifier for facility | 0240d706-3348-5117-8d03-b06c5141e8c0 | yes | Join key |
| hospital_admissions | resident_id | string | Identifier for resident | cf810e7d-ef7d-594d-adb1-4637e10bdd80 | yes | Join key |
| hospital_admissions | effective_date | timestamp | Timestamp for effective date | 2024-12-06 16:53:00 | yes | Start/end validity window |
| hospital_admissions | ineffective_date | timestamp | Timestamp for ineffective date | 2025-01-08 02:30:00 | yes | Start/end validity window |
| hospital_admissions | admission_status | string | Admission status | Chronic Long-Term | yes |  |
| hospital_admissions | emergency_flag | string | Emergency flag |  | yes |  |
| hospital_admissions | hospital_stay_to | timestamp | Hospital stay to | 2024-08-29 00:00:00 | yes |  |
| hospital_admissions | created_at | timestamp | Timestamp for created at | 2024-12-06 17:54:35.743000 | yes | System timestamp |

### hospital_transfers
| Table name | Column name | Data type | Description | Example value | Nullable | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| hospital_transfers | transfer_id | string | Identifier for transfer | 568a2f5c-4bc9-5919-a588-e7e2f944f3cc | yes | Join key |
| hospital_transfers | facility_id | string | Identifier for facility | 0240d706-3348-5117-8d03-b06c5141e8c0 | yes | Join key |
| hospital_transfers | resident_id | string | Identifier for resident | 87db939c-33d3-57c2-bdd0-467a0a3a5b27 | yes | Join key |
| hospital_transfers | effective_date | timestamp | Timestamp for effective date | 2024-11-10 11:30:00 | yes | Start/end validity window |
| hospital_transfers | ineffective_date | timestamp | Timestamp for ineffective date | 2024-11-19 10:29:00 | yes | Start/end validity window |
| hospital_transfers | to_from_type | string | To from type | Acute care hospital | yes |  |
| hospital_transfers | transfer_outcome | string | Transfer outcome | Admitted, Inpatient | yes |  |
| hospital_transfers | stay_purpose | string | Stay purpose | Chronic Long-Term | yes |  |
| hospital_transfers | transfer_reason | string | Transfer reason | Nausea/vomiting | yes |  |
| hospital_transfers | planned_flag | boolean | Planned flag | False | yes |  |
| hospital_transfers | emergency_flag | boolean | Emergency flag | False | yes |  |
| hospital_transfers | created_at | timestamp | Timestamp for created at | 2024-11-11 08:53:38.333000 | yes | System timestamp |

### incidents
| Table name | Column name | Data type | Description | Example value | Nullable | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| incidents | incident_id | string | Identifier for incident | 0d32dafc-e91c-5360-aefd-55e0f98943d6 | yes | Join key |
| incidents | facility_id | string | Identifier for facility | 0240d706-3348-5117-8d03-b06c5141e8c0 | yes | Join key |
| incidents | resident_id | string | Identifier for resident | e681d7c1-a733-57cf-86cd-fd57e3b02f39 | yes | Join key |
| incidents | incident_type | string | Incident type | Fall | yes |  |
| incidents | incident_location | string | Incident location | Resident's Room | yes |  |
| incidents | occurred_at | timestamp | Timestamp for occurred at | 2024-11-13 05:10:00 | yes |  |
| incidents | strikeout | boolean | Strikeout | False | yes |  |
| incidents | created_at | timestamp | Timestamp for created at | 2024-11-13 05:45:31.053000 | yes | System timestamp |

### injuries
| Table name | Column name | Data type | Description | Example value | Nullable | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| injuries | injury_id | string | Identifier for injury | ecb65a51-6852-57fa-9623-ef2e442b4260 | yes | Join key |
| injuries | incident_id | string | Identifier for incident | 0d32dafc-e91c-5360-aefd-55e0f98943d6 | yes | Join key |
| injuries | facility_id | string | Identifier for facility | 0240d706-3348-5117-8d03-b06c5141e8c0 | yes | Join key |
| injuries | injury_type | string | Injury type | Bruise | yes |  |
| injuries | injury_location | string | Injury location | 4) Face | yes |  |
| injuries | is_post_incident | boolean | Boolean flag for is post incident | False | yes |  |
| injuries | created_at | timestamp | Timestamp for created at | 2026-01-15 19:04:08.987000 | yes | System timestamp |

### lab_reports
| Table name | Column name | Data type | Description | Example value | Nullable | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| lab_reports | lab_report_id | string | Identifier for lab report | 0fe8d9f9-bb86-53f6-8803-a0bbf976f34d | yes | Join key |
| lab_reports | resident_id | string | Identifier for resident | 725efbae-bafc-53a6-bfa5-ff3ad771f447 | yes | Join key |
| lab_reports | facility_id | string | Identifier for facility | 0240d706-3348-5117-8d03-b06c5141e8c0 | yes | Join key |
| lab_reports | lab_name | string | Lab name | Urinalysis | yes |  |
| lab_reports | status | string | Status | Completed | yes |  |
| lab_reports | severity_status | string | Severity status | Abnormal | yes |  |
| lab_reports | reported_at | timestamp | Timestamp for reported at | 2024-07-31 00:00:00 | yes |  |
| lab_reports | collected_at | timestamp | Timestamp for collected at | 2024-07-30 00:00:00 | yes |  |
| lab_reports | created_at | timestamp | Timestamp for created at | 2024-08-01 12:38:33.937000 | yes | System timestamp |

### medications
| Table name | Column name | Data type | Description | Example value | Nullable | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| medications | medication_id | string | Identifier for medication | d41edf5d-7d5c-5c39-b8d2-26514cbb1432 | yes | Join key |
| medications | resident_id | string | Identifier for resident | 072691ff-60ba-5d4c-a8b4-ba582a99d803 | yes | Join key |
| medications | facility_id | string | Identifier for facility | 080d2f83-a504-5500-a4ad-1efa311b6fef | yes | Join key |
| medications | description | string | Description | Torsemide Tablet Give 10 mg by mouth in the morning related to ACUTE ON CHRON... | yes |  |
| medications | scheduled_at | timestamp | Timestamp for scheduled at | 2024-06-19 07:00:00 | yes |  |
| medications | administered_at | timestamp | Timestamp for administered at | 2024-06-19 09:21:00 | yes |  |
| medications | status | string | Status | Late | yes |  |
| medications | created_at | timestamp | Timestamp for created at | 2024-06-20 09:04:41.515000 | yes | System timestamp |

### needs
| Table name | Column name | Data type | Description | Example value | Nullable | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| needs | need_id | string | Identifier for need | 4ad8c623-3f7c-5fd8-98fb-f145efba5b54 | yes | Join key |
| needs | resident_id | string | Identifier for resident | a13b0271-693e-589a-aac3-8cdbc200229b | yes | Join key |
| needs | facility_id | string | Identifier for facility | 018d5b79-b86a-5dc8-8453-417d99a8f1f9 | yes | Join key |
| needs | care_plan_id | string | Identifier for care plan | 2e1ab86e-0eb6-5c01-a110-e62d6e9625d0 | yes | Join key |
| needs | need_type | string | Need type | PASSR | yes |  |
| needs | need_category | string | Need category | Other | yes |  |
| needs | initiated_at | timestamp | Timestamp for initiated at | 2025-07-08 00:00:00 | yes |  |
| needs | resolved_at | timestamp | Timestamp for resolved at | 2025-06-17 15:00:00 | yes |  |
| needs | strikeout | boolean | Strikeout | False | yes |  |
| needs | current_row | boolean | Current row | False | yes |  |
| needs | created_at | timestamp | Timestamp for created at | 2025-07-08 13:26:53.020000 | yes | System timestamp |

### physician_orders
| Table name | Column name | Data type | Description | Example value | Nullable | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| physician_orders | order_id | string | Identifier for order | 00ff7cf2-ba99-5c49-bab1-e92b92df41cf | yes | Join key |
| physician_orders | resident_id | string | Identifier for resident | 38f1a419-a8ee-5c49-93be-072c5717e6ed | yes | Join key |
| physician_orders | facility_id | string | Identifier for facility | 0240d706-3348-5117-8d03-b06c5141e8c0 | yes | Join key |
| physician_orders | category | string | Category | Other | yes |  |
| physician_orders | ordered_at | timestamp | Timestamp for ordered at | 2025-01-22 13:00:23.020000 | yes |  |
| physician_orders | start_at | timestamp | Timestamp for start at | 2025-01-22 13:00:00 | yes |  |
| physician_orders | end_at | timestamp | Timestamp for end at | 2024-11-08 19:29:00 | yes |  |
| physician_orders | order_status | string | Order status | Active | yes |  |
| physician_orders | frequency | string | Frequency |  | yes |  |
| physician_orders | created_at | timestamp | Timestamp for created at | 2025-01-22 13:00:23.050000 | yes | System timestamp |

### residents
| Table name | Column name | Data type | Description | Example value | Nullable | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| residents | resident_id | string | Identifier for resident | 578b8ad7-37c2-52b2-8b9e-9d82891ddf41 | yes | Join key |
| residents | facility_id | string | Identifier for facility | 018d5b79-b86a-5dc8-8453-417d99a8f1f9 | yes | Join key |
| residents | date_of_birth | timestamp | Date of birth | 1936-04-15 00:00:00 | yes |  |
| residents | admission_date | timestamp | Timestamp for admission date | 2017-10-25 23:00:00 | yes |  |
| residents | discharge_date | timestamp | Timestamp for discharge date | 2024-12-20 20:46:00 | yes |  |
| residents | deceased_date | timestamp | Timestamp for deceased date | 2025-02-15 17:57:00 | yes |  |
| residents | outpatient | boolean | Outpatient | False | yes |  |
| residents | created_at | timestamp | Timestamp for created at | 2024-12-17 09:44:08.250000 | yes | System timestamp |
| residents | updated_at | timestamp | Timestamp for updated at | 2026-02-13 04:51:18.257000 | yes | System timestamp |

### therapy_tracks
| Table name | Column name | Data type | Description | Example value | Nullable | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| therapy_tracks | therapy_id | string | Identifier for therapy | aa2beff4-3f50-516e-9c0a-fc95370a6efc | yes | Join key |
| therapy_tracks | facility_id | string | Identifier for facility | 080d2f83-a504-5500-a4ad-1efa311b6fef | yes | Join key |
| therapy_tracks | resident_id | string | Identifier for resident | 8f969d63-558f-55a6-9a65-c77b38de3972 | yes | Join key |
| therapy_tracks | discipline | string | Discipline | OT | yes |  |
| therapy_tracks | start_at | timestamp | Timestamp for start at | 2023-09-07 00:00:00 | yes |  |
| therapy_tracks | end_at | timestamp | Timestamp for end at | 2023-08-25 00:00:00 | yes |  |
| therapy_tracks | created_at | timestamp | Timestamp for created at | 2023-09-07 00:00:00 | yes | System timestamp |

### vitals
| Table name | Column name | Data type | Description | Example value | Nullable | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| vitals | vital_id | string | Identifier for vital | 4bd6b2ec-b186-550f-a1a8-42016b72d6b1 | yes | Join key |
| vitals | resident_id | string | Identifier for resident | 207a6546-03d7-5343-a214-299291a28151 | yes | Join key |
| vitals | facility_id | string | Identifier for facility | 0240d706-3348-5117-8d03-b06c5141e8c0 | yes | Join key |
| vitals | vital_type | string | Vital type | Pain Level | yes |  |
| vitals | value | float | Value | 0.0 | yes |  |
| vitals | dystolic_value | float | Dystolic value | 96.0 | yes |  |
| vitals | measured_at | timestamp | Timestamp for measured at | 2024-12-12 04:31:08 | yes |  |
| vitals | strikeout | boolean | Strikeout | False | yes |  |
| vitals | created_at | timestamp | Timestamp for created at | 2024-12-12 05:31:40.283000 | yes | System timestamp |
