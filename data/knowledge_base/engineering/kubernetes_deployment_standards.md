# Enterprise Kubernetes Deployment & Production Standards

**Document ID:** STD-ENG-2026-008  
**Version:** 3.0  
**Effective Date:** January 15, 2026  
**Target Audience:** DevOps Engineers, Platform Architects, Software Engineers  
**Classification:** Internal Technical Standard  

---

## 1. Overview & Purpose
This standard specifies mandatory infrastructure, resource management, security context, and deployment orchestration requirements for all containerized microservices running on Enterprise production Kubernetes clusters (EKS / GKE). Compliance with these standards is enforced via automated admission controllers (Kyverno / OPA Gatekeeper).

---

## 2. Pod Security & Non-Root Execution
In alignment with CIS Kubernetes Benchmarks and SOC2 compliance mandates:
- **Disallow Root Execution:** All container pods must execute as an unprivileged, non-root user. The `securityContext` block must explicitly declare:
  ```yaml
  securityContext:
    runAsNonRoot: true
    runAsUser: 10001
    runAsGroup: 10001
    allowPrivilegeEscalation: false
    readOnlyRootFilesystem: true
    capabilities:
      drop:
        - ALL
  ```
- **Filesystem Write Restrictions:** Ephemeral write operations must be restricted to mounted `emptyDir` volumes mounted explicitly at `/tmp` or `/app/cache`.
- **Privileged Containers:** Deployments specifying `privileged: true` or mounting host root filesystems (`hostPath`) are blocked by the admission controller.

---

## 3. Resource Allocation & Autoscaling (HPA)

### 3.1 Resource Requests & Limits
Every container specification must declare explicit CPU and Memory requests and limits. Sizing without resource requests causes unbounded scheduling instability and cluster node starvation.

| Service Tier | CPU Request | CPU Limit | Memory Request | Memory Limit | Target Utilization |
|:---|:---|:---|:---|:---|:---|
| **API Gateway / Router** | 500m | 2000m | 512Mi | 2048Mi | 65% CPU |
| **Worker / Ingestion** | 1000m | 4000m | 1024Mi | 4096Mi | 75% CPU |
| **Qdrant Vector Node** | 2000m | 8000m | 4096Mi | 16384Mi | 70% Memory |

### 3.2 Horizontal Pod Autoscaler (HPA)
Deployments serving client-facing HTTP traffic must attach an HPA resource configured with both CPU and latency-based metric triggers:
- **Min Replicas:** 3 pods (distributed across distinct availability zones via `topologySpreadConstraints`).
- **Max Replicas:** 24 pods.
- **Scale-Up Stabilization:** 0 seconds (rapid scale-out during traffic bursts).
- **Scale-Down Stabilization:** 300 seconds (5 minutes cooldown to avoid thrashing).

---

## 4. Health Probes & Zero-Downtime Rolling Updates

### 4.1 Probe Specifications
All applications must expose native HTTP endpoints dedicated to Kubernetes health verification:
```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 15
  periodSeconds: 20
  timeoutSeconds: 3
  failureThreshold: 3

readinessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 5
  periodSeconds: 10
  timeoutSeconds: 2
  failureThreshold: 2
```

### 4.2 Rolling Update Strategy & Graceful Termination
- **Deployment Strategy:** `RollingUpdate` with `maxSurge: 25%` and `maxUnavailable: 0` to ensure zero dropped connections during blue-green updates.
- **PreStop Lifecycle Hook:** Containers must define a `preStop` sleep hook of 10 seconds to allow upstream Envoy endpoints to drain connections before SIGTERM is issued:
  ```yaml
  lifecycle:
    preStop:
      exec:
        command: ["/bin/sh", "-c", "sleep 10"]
  ```
- **Termination Grace Period:** Configured to `terminationGracePeriodSeconds: 45`.

---

## 5. Secret Management & Configuration
- **No Plaintext Secrets:** Plaintext API keys, database credentials, and service tokens must never appear in Git repositories or ConfigMaps.
- **External Secrets Operator (ESO):** All application credentials must be injected dynamically via HashiCorp Vault or AWS Secrets Manager synchronized through `ExternalSecret` custom resources.
