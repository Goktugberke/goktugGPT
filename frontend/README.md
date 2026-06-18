# goktugGPT — Frontend

> ⏳ **Not built yet.** This directory is the placeholder for the web UI.

## Plan

A React single-page app that consumes the backend API
([`../backend/`](../backend/)) — the same way ChatGPT/Claude/Gemini frontends work:

- **Auth** — login/register against Keycloak (via the API gateway)
- **Chat** — chat list sidebar + active conversation
- **Streaming** — consume the inference **SSE** stream (`/api/v1/inference/stream`) for the token-by-token typing effect
- **`<think>` panel** — collapsible reasoning block (the model's chain-of-thought)
- **File upload** — presigned-URL flow to the asset service
- **Settings** — profile, custom instructions, billing/token usage, sessions
- **Dark/light mode**

## Design direction

An **ancient-antiquity / Greco-Roman marble** theme (museum-of-knowledge aesthetic):
marble/ivory surfaces with lapis, gold, and Siena-red accents, inscriptional
typography, a fluted **column** as the sidebar divider. Built with Claude Design;
the high-fidelity prototype + tokens are produced separately and will be wired here.

## Tech (planned)

React + Vite + TypeScript · talks to the API gateway at `http://localhost:8080`.
Streaming is a drop-in `EventSource` against the orchestrator's SSE endpoint.
