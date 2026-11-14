"""Check rendered deployment configuration without contacting a cluster.

Usage: python scripts/check_deployments.py --environment prod
Or: kubectl kustomize deploy/k8s/overlays/prod | python scripts/check_deployments.py --environment prod --manifest -
"""

import argparse
from pathlib import Path
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def check(resources: list[dict], environment: str) -> None:
    objects = {(obj["kind"], obj["metadata"]["name"]): obj for obj in resources}
    require(len(objects) == len(resources), "Duplicate resource kind/name")
    namespace = "trendpilot" if environment == "prod" else "trendpilot-dev"
    for obj in resources:
        if obj["kind"] != "Namespace":
            require(obj["metadata"].get("namespace") == namespace, "Resource escaped its namespace")
    pvc = objects["PersistentVolumeClaim", "trendpilot-storage"]
    policy = pvc["metadata"]["annotations"]["argocd.argoproj.io/sync-options"]
    require(
        {"Prune=false", "Delete=false"} <= set(policy.split(",")),
        "PVC must survive prune and app deletion",
    )
    api = objects["Deployment", "trendpilot-api"]["spec"]["template"]["spec"]
    container = next(item for item in api["containers"] if item["name"] == "api")
    require(
        container["readinessProbe"]["httpGet"]["path"] == "/ready", "Readiness must check inference"
    )
    for name in ("startupProbe", "livenessProbe"):
        require(
            container[name]["httpGet"]["path"] == "/health",
            "Process probes must not depend on a model",
        )
    job = objects["CronJob", "trendpilot-retrain"]["spec"]["jobTemplate"]["spec"]["template"][
        "spec"
    ]
    require(not job.get("initContainers"), "Data refresh must share the training process's lock")
    require(len(job["containers"]) == 1, "Expected one coordinated training process")
    trainer = job["containers"][0]
    require(
        trainer["command"] == ["python", "src/modeling/train.py"], "Unexpected training entry point"
    )
    require("--refresh-data" in trainer["args"], "Retraining must refresh under the writer lock")
    require(
        any(mount["mountPath"] == "/app/data/sample" for mount in trainer["volumeMounts"]),
        "Non-root refresh needs writable sample storage",
    )
    if environment == "dev":
        return
    bootstrap = yaml.safe_load((ROOT / "deploy/k8s/bootstrap/namespace-prod.yaml").read_text())
    require(
        bootstrap["metadata"]["name"] == namespace,
        "Missing independent production namespace bootstrap",
    )
    application = yaml.safe_load((ROOT / "deploy/argocd/apps/trendpilot-prod.yaml").read_text())
    app_policy = application["metadata"]["annotations"]["argocd.argoproj.io/sync-options"]
    require(
        {"Prune=confirm", "Delete=confirm"} <= set(app_policy.split(",")),
        "Parent must require confirmation before deleting production",
    )
    require(
        not any(obj["kind"] == "Namespace" for obj in resources),
        "Production workload must not manage its namespace",
    )
    require(
        not any(obj["kind"] == "Secret" for obj in resources),
        "Production secret values must stay outside rendered Git manifests",
    )
    for name in ("trendpilot-api", "trendpilot-dashboard"):
        pod = objects["Deployment", name]["spec"]["template"]["spec"]
        env = {entry["name"]: entry for entry in pod["containers"][0]["env"]}
        require(env["API_KEY_REQUIRED"]["value"] == "true", "Production auth must fail closed")
    for name in ("trendpilot-dashboard", "trendpilot-api"):
        ingress = objects["Ingress", name]
        annotations = ingress["metadata"]["annotations"]
        require(
            ingress["spec"]["ingressClassName"] == "traefik", "Expected maintained Traefik ingress"
        )
        require(
            annotations["traefik.ingress.kubernetes.io/router.entrypoints"] == "websecure",
            "Application routes must be HTTPS-only",
        )
        require(
            annotations["traefik.ingress.kubernetes.io/router.tls"] == "true",
            "Application TLS must be enabled",
        )
        require(
            annotations["traefik.ingress.kubernetes.io/router.tls.options"]
            == f"{namespace}-strict-tls@kubernetescrd",
            "Missing strict SNI/TLS policy",
        )
        require(bool(ingress["spec"]["tls"][0]["secretName"]), "Missing TLS secret reference")
        require(
            not any(key.startswith("nginx.") for key in annotations),
            "Retired controller annotations remain",
        )
    dashboard = objects["Ingress", "trendpilot-dashboard"]["metadata"]["annotations"]
    require(
        dashboard["traefik.ingress.kubernetes.io/router.middlewares"]
        == f"{namespace}-dashboard-auth@kubernetescrd",
        "Dashboard auth middleware is detached",
    )
    auth = objects["Middleware", "dashboard-auth"]["spec"]["basicAuth"]
    require(
        auth["secret"] == "trendpilot-dashboard-auth" and auth["removeHeader"],
        "Missing dashboard Secret auth",
    )
    redirect = objects["Ingress", "trendpilot-https-redirect"]["metadata"]["annotations"]
    require(
        redirect["traefik.ingress.kubernetes.io/router.entrypoints"] == "web",
        "HTTP route must only redirect",
    )
    require(
        redirect["traefik.ingress.kubernetes.io/router.middlewares"]
        == f"{namespace}-https-redirect@kubernetescrd",
        "Redirect middleware is detached",
    )
    require(
        objects["Middleware", "https-redirect"]["spec"]["redirectScheme"]
        == {"scheme": "https", "permanent": True},
        "HTTP must permanently redirect to HTTPS",
    )
    require(
        objects["TLSOption", "strict-tls"]["spec"]["sniStrict"] is True,
        "Fallback TLS certificate must not be accepted",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", choices=["dev", "prod"], required=True)
    parser.add_argument(
        "--manifest", help="Rendered YAML file, or - for stdin; otherwise run kubectl kustomize"
    )
    args = parser.parse_args()
    if args.manifest == "-":
        content = sys.stdin.read()
    elif args.manifest:
        content = Path(args.manifest).read_text()
    else:
        content = subprocess.run(
            ["kubectl", "kustomize", str(ROOT / "deploy/k8s/overlays" / args.environment)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    resources = [obj for obj in yaml.safe_load_all(content) if obj]
    try:
        check(resources, args.environment)
    except (KeyError, TypeError, ValueError) as exc:
        parser.exit(1, f"Deployment invariant failed: {exc}\n")
    print(f"{args.environment}: {len(resources)} resources; deployment checks passed")


if __name__ == "__main__":
    main()
