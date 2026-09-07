# Enterprise Billing, Refund & Invoice Dispute Policy

**Document ID:** POL-REV-2026-004  
**Version:** 2.8  
**Effective Date:** February 1, 2026  
**Target Audience:** Enterprise Customers, Finance Teams, Billing Operations  
**Department:** Revenue Operations, Corporate Billing & Legal  

---

## 1. Scope & Core Principles
This policy establishes standardized guidelines regarding billing cycles, annual license terms, credit adjustments, and the resolution of commercial payment disputes. All sales of enterprise software subscriptions, professional implementation services, and token consumption overages are governed by these provisions.

---

## 2. 30-Day Commercial Money-Back Guarantee
To provide prospective enterprise clients with confidence during onboarding and evaluation phases:
- **Eligibility Window:** New enterprise customers entering an initial annual Master Services Agreement (MSA) are eligible for a **100% unconditional refund** of base subscription platform fees within the first **30 calendar days** following the contract effective date.
- **Exclusions:**
  - Customized professional onboarding and systems integration fees are non-refundable once hours have been delivered.
  - LLM token consumption overages exceeding contracted baseline tier limits are billed at pass-through cost and excluded from refunds.
- **Execution Workflow:** A written cancellation notice must be submitted by an authorized customer representative to `billing-ops@enterprise-ai.internal` prior to midnight UTC on day 30.

---

## 3. Subscription Cancellations & Mid-Term Renewals
- **Annual Contracts:** Enterprise agreements are multi-month or multi-year commercial commitments. Mid-term cancellations outside the 30-day guarantee period do not qualify for pro-rated cash refunds; subscriptions remain active through the expiration of the prepaid term.
- **Notice of Non-Renewal:** Written notice of intent not to renew must be delivered at least **60 calendar days** prior to the anniversary renewal date.
- **Seat Reductions:** Reductions in contracted seats or tier downgrades take effect only upon contract renewal.

---

## 4. Invoice Dispute Resolution & Credit Adjustments
In the event a customer identifies a billing calculation discrepancy, duplicate charge, or unauthorized usage:
1. **Dispute Filing:** Customers must file a formal dispute ticket via the billing portal within **45 calendar days** of the disputed invoice date, detailing the specific line items and rationale.
2. **Investigation Window:** Enterprise Revenue Operations will review usage logs, API metering telemetry, and billing records, issuing a formal resolution determination within **10 business days**.
3. **Credit Notes vs. Cash Refunds:** Approved adjustments for active subscription accounts are issued as credit notes applied automatically against subsequent invoices. Cash refunds are issued only upon mutual termination of the contract.

---

## 5. Late Payments & Service Suspension Safeguards
- **Payment Terms:** Standard payment terms are Net 30 from the date of invoice issuance.
- **Grace Period & Cure Window:** If payment is not received by day 30, automated payment reminders are triggered. A formal **15-day cure notice** is issued on day 45.
- **Service Suspension:** Production API access may be suspended if an account remains delinquent past **60 calendar days**, provided at least three written notices have been sent to the designated billing contact. Ingested customer knowledge bases and vector indexes are preserved intact for 90 days following suspension.
