---
og:title: "Security considerations"
og:description: "What Tesseract does and doesn't protect against when you build, run, serve, and call Tesseracts."
---

# Security considerations

A Tesseract is code. Building, running, or loading one gives that code the same access as any other program you run. Tesseract doesn't sandbox it, and a served Tesseract doesn't check who is calling it, so access control has to come from the network or a proxy in front of it.

## Don't

- **Don't build Tesseracts from source you don't trust.** `tesseract build` installs whatever the requirements file lists, runs `custom_build_steps`, and pulls whatever `base_image` the config names. Treat it like `docker build` on an untrusted Dockerfile.
- **Don't run images you don't trust.** This covers `tesseract run`, `tesseract serve`, and `Tesseract.from_image`.
- **Don't load a `tesseract_api.py` you don't trust with `Tesseract.from_tesseract_api`.** It's imported into your Python process, with no container in between.
- **Don't serve on a non-loopback address unless something else limits who can connect.** `tesseract serve` binds to `127.0.0.1` by default. With `--host-ip 0.0.0.0`, anyone who can reach the port can call every endpoint. Put a firewall, VPN, or authenticating reverse proxy in front.

## Don't assume

- **That the container is a sandbox.** It's an ordinary Docker container, run as your user with host directories mounted. If you need isolation, harden it yourself.
- **That a served Tesseract is safe from its clients.** The runtime validates requests against the Tesseract's schemas, and that's all it does. It doesn't limit request size or rate. A `json+binref` input can point at an absolute path or a URL, so a client can make the server read any file the container can access, or fetch any URL the server can reach. Code in `tesseract_api.py` that interprets inputs (paths, expressions, shell commands) is exposed to every client.
- **That a server is a Tesseract because it looks like one.** Any server can imitate the Tesseract API, and `Tesseract.from_url` can't tell the difference. Treat what a server returns like any other untrusted HTTP response, and only connect to servers you trust.
