# Enterprise Procurement & Spend Authorization Matrix

**Document ID:** SOP-FIN-2026-005  
**Version:** 3.2  
**Effective Date:** February 1, 2026  
**Target Audience:** Budget Owners, Department Leads, Executive Leadership  
**Department:** Corporate Procurement, Strategic Sourcing & Legal  

---

## 1. Overview & Governance Framework
This standard operating procedure defines the financial authorization thresholds, competitive bidding mandates, and contract approval workflows for all corporate expenditures, SaaS software acquisitions, professional consulting contracts, and capital equipment purchases.

---

## 2. Spend Authorization & Approval Thresholds

All purchase orders (PO), statements of work (SOW), and vendor contracts require formal sign-off in the Coupa procurement portal prior to vendor commitment. Approvals follow a tiered delegation of authority:

| Annual Contract Value (USD) | Required Approvers | Bidding Requirement | Security Review Mandatory? |
|:---|:---|:---|:---|
| **$0 - $5,000** | Hiring Manager / Direct Manager | 1 Quote | If processing customer data |
| **$5,001 - $25,000** | Department Director + Finance Analyst | 2 Comparable Quotes | Yes (InfoSec Tier 3) |
| **$25,001 - $100,000** | Functional Vice President + Head of Procurement | 3 Competitive RFP Bids | Yes (InfoSec Tier 2) |
| **$100,001 - $500,000** | C-Level Executive (CTO/CRO/CPO) + CFO | Formal RFP Process | Yes (InfoSec Tier 1 + SOC2) |
| **> $500,000** | Chief Executive Officer (CEO) + Board of Directors Audit Committee | Full Strategic Sourcing RFP | Full Architecture & Legal Audit |

---

## 3. Mandatory Information Security & Legal Vetting Triggers
Regardless of monetary contract value, formal Information Security Risk Assessment (InfoSec) and Legal Counsel review are automatically triggered if any of the following conditions are met:
1. **Customer PII or Telemetry Processing:** The vendor software or cloud service will store, process, transmit, or index customer confidential data or personal identifiers.
2. **Production Code or Infrastructure Access:** The software requires read/write access to internal GitHub repositories, AWS IAM credentials, or Kubernetes clusters.
3. **AI / Machine Learning Foundation Model Inference:** The vendor provides generative LLM capabilities where customer inputs might be used for model training or retained outside defined enterprise boundaries.
4. **Auto-Renewal Clauses:** Contracts featuring automatic evergreen renewal clauses with notice windows exceeding 30 days must be struck down by Corporate Legal.

---

## 4. Software License Management & Shadow IT Prohibitions
- **Shadow IT Zero-Tolerance:** Employees and managers are strictly prohibited from expensing recurring software subscriptions (e.g., Notion, ChatGPT Plus, GitHub Copilot) via corporate credit cards without an authorized purchase order and InfoSec clearance.
- **Centralized License Depository:** All approved SaaS licenses are managed and provisioned centrally by Enterprise IT through Okta Single Sign-On (SSO).
- **Periodic License Audits:** Procurement conducts quarterly utilization reviews. Unassigned seats or subscriptions inactive for 60 consecutive days are automatically de-provisioned and reallocated.

---

## 5. Emergency Procurement Escalations
In the event of an active P1 production outage where immediate vendor assistance or infrastructure expansion is required to prevent catastrophic business disruption:
- The Incident Commander may authorize emergency spend up to **$25,000 USD** with verbal/Slack sign-off from an on-call Engineering Director.
- Retroactive Coupa PO documentation and formal CFO notifications must be finalized within **2 business days** following incident closure.
