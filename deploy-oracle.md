# Oracle interview demo

Build the Dockerfile and run a single worker, with a persistent directory owned by UID 10001 mounted at `/state`. Set `GROQ_API_KEY` privately; do not include it in the image. Join the existing `portfolio-edge` Docker network with network alias `aster-api`. Do not publish the backend port.

Set `FORWARDED_ALLOW_IPS=172.19.0.2` to trust proxy headers only from the existing Caddy container, and launch Uvicorn with proxy header support (its default). The Caddy virtual host is:

```caddy
aster-161-118-164-31.sslip.io {
    encode zstd gzip
    request_body { max_size 8KB }
    reverse_proxy aster-api:8000
}
```

Use `--memory 1g --cpus 1 --restart unless-stopped`. Do not run multiple workers; session ownership and per-client rate counters are process-local. Daily inference allowance is stored on the persistent volume. `/api/health` reports liveness only; verify a real fictional policy answer separately before announcing deployment readiness.
