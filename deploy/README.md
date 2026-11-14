# Kubernetes and GitOps deployment

Kustomize composes the API, dashboard, storage, and retraining workloads. Argo CD
manages the environment Applications. Production uses Traefik v3 for HTTPS,
HTTP redirects, dashboard basic auth, and WebSocket forwarding.

## Layout and ownership

| Path | Owner and purpose |
|---|---|
| `k8s/base/` | Shared API/dashboard Deployments and Services, PVC, ConfigMap, CronJob |
| `k8s/components/postgres/` | Optional development PostgreSQL and ingress policy |
| `k8s/overlays/dev/` | Local images, disposable DB credentials, suspended retraining |
| `k8s/overlays/prod/` | GHCR images, RWX storage, 2 API replicas, Traefik routes, PDB |
| `k8s/bootstrap/namespace-prod.yaml` | Administrator-owned production namespace; outside the workload app |
| `traefik/values.yaml` | Administrator-owned controller prerequisites; outside the workload app |
| `argocd/apps/` | Workload AppProject and dev/prod Applications |

The production workload does not own its Namespace, ingress controller, CRDs,
TLS certificate issuer, or secret manager. Bootstrap these separately. The dev
Application still owns its local namespace, with deletion confirmation enabled.

## Offline checks

After `uv sync --locked --all-extras`, with kubectl installed:

```bash
uv run --no-sync python scripts/check_deployments.py --environment dev
uv run --no-sync python scripts/check_deployments.py --environment prod
```

These commands only run `kubectl kustomize` and inspect the result. They check
readiness, writer coordination, retention, auth, TLS, and middleware references;
they do not apply resources or perform complete Kubernetes schema validation.

The controller smoke test uses Docker, kubectl, openssl, and the API image built
by `make build`. It creates disposable test credentials and removes its containers
and fixtures afterward:

```bash
python3 scripts/smoke_ingress.py --app-image trendpilot-ai-api:latest
```

It translates the rendered policies to Traefik's file provider and checks both
HTTP redirects, anonymous/incorrect-password rejection, authenticated requests,
Authorization-header removal, WebSocket upgrade, and missing auth/certificate
failure. It does not simulate Kubernetes Secret watches, CRD admission, or Argo
reconciliation. Run the cluster acceptance checks below before public exposure.

## Storage and model publication

Base storage is ReadWriteOnce. Dashboard and training affinity matches scheduled
API pods; it does not require a Ready API endpoint. Production uses ReadWriteMany.
Choose a StorageClass/backend supporting atomic same-filesystem rename and POSIX
advisory locks **across nodes**. All writers must use the same UID/GID and volume.
Validate these properties on the actual CSI/backend; RWX by itself is insufficient.

Set `storageClassName` in `overlays/prod/storage-patch.yaml`. Production provisioning
must also use a PV reclaim policy of `Retain` and a tested external backup/restore
procedure. No portable default StorageClass can establish those guarantees for an
unknown cluster. Verify them before loading valuable data.

The shared PVC has `Prune=false,Delete=false`, protecting it from both Argo sync
pruning and Application cascading deletion. The prod child Application has
`Prune=confirm,Delete=confirm`, requiring approval before its parent removes it.
These controls do not prevent an administrator from directly deleting a namespace,
PVC, or PV. Namespace deletion removes namespaced resources regardless of PVC
annotations. Treat direct namespace deletion as destructive.

Training holds `models/.training.lock` across data refresh, preparation, training,
validation, and publication. Dashboard/CLI training, CronJobs, manually cloned Jobs,
and standalone ingestion CLI use that same lock. Contention fails clearly rather
than starting another writer. Process exit releases the lock; never delete the
lock file to break a lock, since that creates a second lock inode.

New artifacts have this layout:

```text
models/
├── .training.lock
├── current.json                 # atomically replaced active bundle ID
├── bundles/<bundle-id>/
│   ├── trend_model.keras
│   ├── preprocessor.joblib
│   ├── model_metadata.json
│   └── manifest.json             # SHA-256 for the three artifacts
└── metadata/tuning/              # optional tuning inputs
```

Checkpoints stay in a private staging directory. A serialized prediction probe,
metadata checks, and content hashes must pass before promotion. The bundle
directory is renamed on the same filesystem, then `current.json` is replaced
atomically. Published bundles are not overwritten or automatically garbage-collected.
Interrupted staging leaves the previous pointer intact; a failure after sealing a
bundle can leave an unreferenced bundle, which is safe to retain.

The API validates and loads candidates in a background thread and swaps one complete
model/preprocessor/metadata snapshot. Requests already using a snapshot keep it.
Failed reloads retain the last working snapshot and set `reload_failed` in `/health`.
Prior bundles remain available for a deliberate rollback. Any rollback must select
an entire previously verified bundle by atomically replacing the pointer while
holding the writer lock; never copy individual files over an active bundle.
Backups should capture bundles and their active pointer consistently.

Legacy `models/saved/trend_model.keras` plus `models/metadata/{preprocessor.joblib,
model_metadata.json}` remain readable when no pointer exists. The first successful
new training run publishes a bundle without overwriting those legacy files. Deploy
the updated API, dashboard, and training image together: older releases do not
follow `current.json`.

## Liveness, readiness, and initial training

`/health` reports process liveness without waiting for TensorFlow/model loading.
Startup and liveness probes use it. `/ready` returns 503 until a validated complete
model is loaded, then 200 with the serving model version. Both probe endpoints are
public, but invalid required API-key configuration returns 503 on them as well.

Readiness stays 200 during replacement while the previous validated model can
serve. Kubernetes uses `/ready` to avoid routing inference traffic to an empty or
failed replacement pod. A cold cluster remains live but unready until first training.

The dashboard can start independently of API readiness and trains from the shared
volume. CronJobs also need no API endpoint. Start the first job explicitly; pod
readiness probes contact pod IPs directly, so the finished bundle can be loaded
before any API Service endpoints become Ready.

## Development cluster

Build and load `trendpilot-api:dev` and `trendpilot-dashboard:dev` into the local
cluster's image store (for kind, use `kind load docker-image`). Then:

```bash
kubectl apply -k deploy/k8s/overlays/dev
kubectl -n trendpilot-dev create job --from=cronjob/trendpilot-retrain trendpilot-retrain-init
kubectl -n trendpilot-dev port-forward svc/trendpilot-dashboard 9174:8501
```

The dev CronJob is suspended and uses state-level data. Dev API auth remains
optional and its PostgreSQL credentials are disposable. Keep this overlay private.
Application pods use UID/GID 1000 with dropped capabilities and writable `HOME=/tmp`;
the PostgreSQL component uses UID/GID 70. The API storage initializer creates raw,
processed, sample, and model subdirectories. Retraining mounts writable sample
storage so non-root ingestion does not attempt to overwrite image-owned files.

## Fresh production bootstrap

1. Choose a maintained, reviewed Traefik v3 chart/image and matching CRDs. Pin their
   versions in your administrative infrastructure configuration and use
   `traefik/values.yaml`. Both Kubernetes Ingress and CRD providers must be enabled;
   class `traefik`, entrypoints `web` (80) and `websecure` (443), and the legacy
   namespace-name reference convention (`safeNaming: false`) must match this overlay.
   Keep cross-namespace references and Traefik's own dashboard exposure disabled.
2. Create the namespace separately:

   ```bash
   kubectl apply -f deploy/k8s/bootstrap/namespace-prod.yaml
   ```

3. Provision RWX storage with locking/rename support, a Retain reclaim policy, and
   backups. Publish reviewed API/dashboard images to GHCR and set their release
   identifiers in `overlays/prod/kustomization.yaml`. CI currently builds only.
4. Replace both example hosts in `ingress.yaml`. If renaming namespace `trendpilot`,
   update the namespace-qualified middleware/TLS-option annotations too.
5. Provision `trendpilot-tls` with a valid certificate covering **both** hostnames.
   The `strict-tls` TLSOption requires TLS 1.2+ and strict SNI, rejecting an unmatched
   fallback certificate. Ensure Traefik has permission to read the referenced Secrets.
6. Provision an independent production API key from an external file or secret manager:

   ```bash
   TRENDPILOT_API_KEY_FILE="$HOME/.config/trendpilot-ai-prod/api-key" python3 scripts/api_key.py init
   kubectl -n trendpilot create secret generic trendpilot-secrets \
     --from-file=API_KEY="$HOME/.config/trendpilot-ai-prod/api-key"
   ```

   The API and dashboard require a nonempty `API_KEY`; optional `DATABASE_URL` belongs
   in the same Secret. Do not save literal credentials or rendered Secret YAML here.
7. Create the dashboard authentication Secret using **bcrypt** hashes under
   key **`users`**, as required by the Traefik BasicAuth middleware:

   ```bash
   htpasswd -Bc "$HOME/.config/trendpilot-ai-prod/dashboard-auth" admin
   chmod 600 "$HOME/.config/trendpilot-ai-prod/dashboard-auth"
   kubectl -n trendpilot create secret generic trendpilot-dashboard-auth \
     --from-file=users="$HOME/.config/trendpilot-ai-prod/dashboard-auth"
   ```

   The `dashboard-auth` Middleware removes Authorization before forwarding. Missing
   middleware/credentials must deny routing; do not add an unauthenticated fallback.
   Traefik handles WebSocket upgrades without nginx-specific annotations.
8. Add image pull credentials if GHCR packages are private. Install Argo CD separately,
   apply `argocd/root-app.yaml`, review the production diff, and manually sync prod.
   Bootstrap the first model without waiting for API readiness:

   ```bash
   kubectl -n trendpilot create job --from=cronjob/trendpilot-retrain trendpilot-retrain-init
   ```

Argo CD reads the configured Git revision. Rendering and local checks do not
apply resources or update the deployment.

## Changing namespace and ingress ownership

If an existing production Namespace is tracked by a workload Application, protect
it and the PVC before removing the Namespace from that Application's manifests:

```bash
kubectl annotate namespace trendpilot argocd.argoproj.io/sync-options='Prune=false,Delete=false' --overwrite
kubectl -n trendpilot annotate pvc trendpilot-storage argocd.argoproj.io/sync-options='Prune=false,Delete=false' --overwrite
kubectl apply -f deploy/k8s/bootstrap/namespace-prod.yaml
```

With prod sync paused/manual, remove **only that Namespace's** Argo tracking
metadata according to your installation's tracking mode. For the standard
annotation/label modes, inspect and remove `argocd.argoproj.io/tracking-id` and
`app.kubernetes.io/instance` when they identify `trendpilot-prod`. Custom tracking
label keys require the corresponding adjustment. Never remove PVC tracking.
Verify the production diff has no Namespace deletion and the parent diff preserves
the child Application's confirmation policy before syncing.

When changing ingress controllers, install Traefik and its CRDs, update the
`users` key on the dashboard-auth Secret, and verify the protected HTTPS route
before switching traffic. Retain the old routes until the cutover is complete.
For a parallel cutover, create temporary Traefik Ingress copies with distinct names
and retain the old controller's routes until traffic has moved. Validate the new
load-balancer address using the intended Host/SNI before switching DNS. Updating
the same existing Ingress names in place changes controller ownership and can
interrupt traffic; use a maintenance window if parallel routes are unavailable.
To update the existing auth Secret without writing its contents into the checkout:

```bash
kubectl -n trendpilot create secret generic trendpilot-dashboard-auth \
  --from-file=users="$HOME/.config/trendpilot-ai-prod/dashboard-auth" \
  --dry-run=client -o yaml | kubectl apply -f -
```

API and writer images must both support the bundle layout before the first
bundle is published. Changes to namespace ownership, controller resources, and
Secrets require their own deployment steps.

## Cluster acceptance and recovery checks

In a disposable cluster with the actual CNI/CSI/controller and certificate manager:

- Check HTTP-to-HTTPS redirects on both names; anonymous and wrong-password dashboard
  requests must fail; authenticated HTTP and WebSocket sessions must work.
- Remove auth/TLS references in disposable test resources and confirm failure without
  a public fallback. Check both certificate names and controller/Secret watch behavior.
- Confirm cold boot is live/unready, first training makes API endpoints Ready, and a
  rolling API replacement never becomes Ready before loading a validated bundle.
- Run simultaneous writers on separate nodes, interrupt a writer, and corrupt a new
  bundle. Verify exclusion, release after process death, and retention of the working
  model during failed reload. Confirm the StorageClass's actual lock/rename semantics.
- Rehearse root/child Application deletion and pruning. Confirm namespace isolation and
  PVC retention; check PV reclaim policy and restore a consistent external backup.

API-key rotation still requires coordinated API/dashboard pod replacement because
these manifests inject Secret values as environment variables. Use distinct keys
per environment and an external secret manager for ongoing production operation.

## Deployment constraints

- The production Application tracks `main`, and the example images use mutable
  tags. Manual sync controls timing, not the release contents: select a reviewed
  Git revision and immutable image digests when promoting a release.
- The root Application uses Argo CD's `default` project and manages child
  Applications/AppProjects. Restrict its source and administration to trusted
  maintainers. Dev and prod share a workload AppProject; its Namespace permission
  is needed for the dev overlay.
- Only PostgreSQL has a NetworkPolicy. The dashboard/API Services are reachable
  within the cluster where networking permits, and direct dashboard access bypasses
  ingress basic auth. Use an appropriate enforcing CNI and network isolation before
  exposing this deployment to untrusted workloads. Application pods currently use
  default service accounts with token automounting.
- Secrets are provisioned separately and injected as environment variables. Keep
  access limited, configure encryption at rest in the cluster, and use only the
  intended keys in `trendpilot-secrets`; the API imports that Secret with `envFrom`.
  Secret base64 encoding is not encryption. Single-key rotation needs coordinated
  pod replacement; overlapping old/new credentials are not supported.
- Dashboard training runs in its pod alongside UI sessions. Keep one dashboard
  replica and account for training memory in its resource limit. The writer lock
  serializes training but is not a job queue. Dashboard/training affinity still
  colocates them with an API pod even with production RWX storage; API spread is
  a scheduling preference rather than a hard availability guarantee.
- CI runs Python checks/tests, both image builds, rendered overlay checks, and
  the disposable Traefik smoke test. It does not publish images, validate full
  Kubernetes/CRD schemas, or test cluster reconciliation. Publishing and deployment
  remain separate. The local `:dev` images must be rebuilt, loaded into the cluster,
  and their pods replaced when code changes.

A rendered overlay or local proxy smoke test cannot establish CNI enforcement,
CSI locking/rename semantics, Kubernetes Secret watches, Argo deletion behavior,
or backup recovery. Use the cluster acceptance checks above for those guarantees.

References: [Traefik Ingress](https://doc.traefik.io/traefik/reference/routing-configuration/kubernetes/ingress/),
[BasicAuth Secrets](https://doc.traefik.io/traefik/reference/routing-configuration/http/middlewares/basicauth/),
[TLS options](https://doc.traefik.io/traefik/reference/routing-configuration/http/tls/tls-options/),
[Argo deletion options](https://argo-cd.readthedocs.io/en/stable/user-guide/sync-options/#no-resource-deletion).
