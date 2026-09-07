# Engineering Production Incident Response Runbook

**Document ID:** RUN-ENG-2026-004  
**Version:** 4.2  
**Effective Date:** March 1, 2026  
**Target Audience:** Site Reliability Engineers (SRE), Platform Teams, On-Call Engineers  
**Classification:** Internal Confidential  

---

## 1. Incident Severity Definitions & Response SLAs

| Severity | Definition | Initial Response SLA | Status Page Update | Post-Mortem Required? |
|:---|:---|:---|:---|:---|
| **P1 - Critical** | Complete service outage or critical data degradation affecting > 10% of active enterprise customers. Total loss of core functionality (e.g., API gateway down, database corruption, token authentication failure). | **15 minutes** | Every 20 minutes | **Yes (Mandatory within 48h)** |
| **P2 - Major** | Significant performance degradation, elevated error rates (5xx > 2%), or critical sub-system failure with an active workaround available. Core service remains operational. | **30 minutes** | Every 45 minutes | **Yes (Within 5 business days)** |
| **P3 - Moderate** | Minor feature failure or non-critical latency spike affecting isolated customer cohorts (< 2%). Workaround readily available. | **2 hours** | As needed | Optional |
| **P4 - Low** | Cosmetic bug, minor telemetry failure, or internal development environment disruption. | **Next business day** | None | No |

---

## 2. Command Hierarchy & Incident Roles
During any active P1 or P2 incident, normal organizational hierarchy is suspended. Operational command shifts to the following defined roles:

### 2.1 Incident Commander (IC)
- Owns overall operational coordination, task allocation, and decision authority.
- Designates a dedicated Zoom/Slack war room: `#incident-{YYYYMMDD}-{brief-title}`.
- Declares operational states: Triage $\to$ Mitigating $\to$ Resolved $\to$ Closed.
- Authorizes emergency rollbacks, traffic draining, or feature flag disabling.

### 2.2 Communications Lead (CL)
- Responsible for all outward-facing stakeholder communications.
- Updates the external status page (`status.enterprise-ai.internal`) within the required SLA intervals.
- Coordinates executive briefing with the VP of Engineering and Customer Support Director.

### 2.3 Operations Lead (Ops)
- Directs investigation into root causes, infrastructure telemetry, and log correlation.
- Coordinates deployment of hotfixes, pod restarts, or database failovers.
- Ensures all commands executed in production are logged in the incident scratchpad.

---

## 3. Immediate Triage & Mitigation Workflows

### 3.1 Step 1: Verification & Escalation (0 - 15 Minutes)
1. Acknowledge the PagerDuty alert immediately:
   ```bash
   pagerduty-cli ack --incident-id <ID>
   ```
2. Inspect primary Datadog and Grafana dashboards for ingress error rates and latency anomalies:
   - **Ingress 5xx Spike:** Check Cloudflare / Envoy Gateway edge metrics.
   - **Vector DB Latency:** Check Qdrant RPC response durations and CPU throttling.
   - **Upstream LLM Timeout:** Check Gemini / OpenAI API failure rates and rate limit exhaustion.
3. If confirmed P1, page the secondary on-call engineer and notify `#incident-alerts`.

### 3.2 Step 2: Mitigation Protocols (Stop the Bleeding)
*Priority is mitigation over root-cause investigation.*
- **Automated Canary Rollback:** If the incident coincided with an ArgoCD continuous deployment, trigger immediate canary rollback:
  ```bash
  argocd app rollback enterprise-agent-prod --to-revision <PREVIOUS_HEALTHY_REVISION>
  ```
- **Circuit Breaker Activation:** If an upstream dependency (e.g., third-party LLM API) is failing intermittently, trip the circuit breaker to fall back to the secondary model or cached responses:
  ```bash
  curl -X POST http://internal-admin.enterprise.corp/api/circuit-breakers/llm-primary/trip \
       -H "Authorization: Bearer $EMERGENCY_ADMIN_TOKEN"
  ```
- **Traffic Shedding:** Under extreme database saturation, enable sliding-window shed controls to protect healthcheck routes:
  ```bash
  kubectl patch deployment enterprise-agent-app -n prod -p '{"spec":{"replicas":16}}'
  ```

---

## 4. Blameless Post-Mortem Guidelines
Every P1 and P2 incident requires a blameless post-mortem meeting within 48 hours of resolution.
- **Focus:** Systems, monitoring gaps, automation limitations, and recovery mechanisms—never individual human culpability.
- **Deliverable:** Markdown document in Confluence following the standard template:
  1. Executive Summary & Impact Analysis.
  2. Complete Chronological Timeline (UTC).
  3. Root Cause Analysis (5 Whys methodology).
  4. What Went Well vs. Where We Got Lucky.
  5. Action Items with assigned owners and Jira ticket links (P0 action items must be completed within 14 calendar days).
