# GDPR & Global Data Privacy Governance Policy

**Document ID:** POL-LEGAL-2026-003  
**Version:** 3.1  
**Effective Date:** February 15, 2026  
**Applicability:** Global Customer Data, EU/UK Data Subjects, Enterprise Operations  
**Department:** Legal, Compliance & Data Protection Office (DPO)  

---

## 1. Regulatory Context & Commitment
The Enterprise adheres strictly to the General Data Protection Regulation (Regulation (EU) 2016/679 - GDPR), the UK Data Protection Act 2018, and the California Consumer Privacy Act (CCPA/CPRA). This policy defines our obligations regarding the lawful processing of Personally Identifiable Information (PII), data subject rights, and cross-border data transfer safeguards.

---

## 2. Core Privacy Principles
All engineering architecture and agentic processing must satisfy the principles of **Privacy by Design and by Default**:
1. **Lawfulness, Fairness & Transparency:** Data is processed solely pursuant to legitimate contractual execution, user consent, or explicit legal obligations.
2. **Purpose Limitation:** Customer queries, vector embeddings, and ingested documents may not be re-used for training foundational public models or unauthorized marketing.
3. **Data Minimization:** Only data strictly necessary for processing AI queries and generating verified citations may be ingested and stored.
4. **Accuracy:** Systems must provide mechanisms for data subjects to rectify outdated personal records.
5. **Storage Limitation:** Customer telemetry and conversation transcripts are automatically purged in accordance with defined data retention schedules.

---

## 3. Data Subject Rights & Fulfillment Protocols

### 3.1 Data Subject Access Requests (DSAR)
- **Fulfillment SLA:** All verified DSAR requests must be fulfilled within **30 calendar days** from initial receipt.
- **Scope:** Complete export of all personal data held across vector stores, relational databases, audit logs, and customer support transcripts provided in a machine-readable format (JSON or CSV).

### 3.2 Right to Erasure ("Right to be Forgotten")
Upon receipt of a verified deletion request:
- **Relational Databases:** Scrub or hard-delete all direct identifiers (`customer_id`, email, names) within 72 hours.
- **Vector Stores & Embeddings:** Execute deterministic deletion of chunk points referencing the customer document ID:
  ```python
  await vector_store.delete_by_document_id(document_id=target_id)
  ```
- **Backup Exceptions:** Cryptographically signed cold backups are overwritten through standard lifecycle roll-off within 30 days.

---

## 4. Regional Data Residency & Sovereignty Architecture
- **EU Data Subjects:** All production clusters, Qdrant vector collections, and relational databases processing personal data belonging to EU citizens must reside exclusively within the **AWS `eu-west-1` (Ireland)** or **GCP `europe-west1` (Belgium)** regions.
- **No Cross-Border Egress:** Personal data originating within the European Economic Area (EEA) may not transit into US-based inference endpoints unless bound by Standard Contractual Clauses (SCC) and encrypted with customer-managed keys (BYOK).

---

## 5. Security Incident & Breach Notification SLA
In the event of a confirmed or suspected security compromise involving personal data:
- **Notification to Data Protection Authorities (DPA):** The Enterprise Data Protection Officer (DPO) must report the breach to the relevant supervisory authority (e.g., the Irish Data Protection Commission) within **72 hours** of becoming aware of the breach.
- **Affected Data Subjects:** If the breach poses a high risk to individual privacy and rights, affected customers must be notified without undue delay via electronic mail and public incident disclosure.

---

## 6. Data Protection Officer Contact
For regulatory notifications, supervisory audits, or escalated DSAR inquiries:
- **DPO Email:** `dpo@enterprise-ai.internal`
- **Postal Address:** Enterprise AI Data Protection Office, 100 Privacy Boulevard, Dublin 2, Ireland.
