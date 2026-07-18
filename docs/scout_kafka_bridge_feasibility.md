# Feasibility study: a JPL Scout → Kafka bridge for Rubin ToO alerting

*Status: proposal (2026-07-17). Companion prototype plan in §10–§12.*

## Summary

We assess the feasibility of a containerized service that polls the JPL/CNEOS **Scout**
NEOCP hazard-assessment system and republishes new, updated, and cancelled
Rubin-ToO-candidate objects as a **Kafka stream** consumable by the Vera C. Rubin
Observatory's Target-of-Opportunity system.

**Verdict: feasible, inexpensive, and well-timed.** Rubin's first-year ToO operations
paper ([arXiv:2607.00217](https://arxiv.org/html/2607.00217v1)) states that *"no kafka
stream for potentially hazardous asteroids is available"* and explicitly encourages the
publication of machine-readable Scout alerts to a Kafka stream. Nearly all of the domain
logic required already exists in [`tom_jpl`](https://github.com/TOMToolkit/tom_jpl)
(Scout ingestion and change reconciliation) and in FOMO's `solsys_code/rubin_too.py`
(the SSSC NEOs WG ToO filter criteria). The remaining work is a Kafka publisher, a
container, and deployment plumbing.

## 1. Background

Through its ToO program, Rubin reserves a fraction of survey time for rare,
time-sensitive events. The ToO alert-ingestion pipeline consumes external alerts
(currently LVK gravitational-wave and IceCube neutrino streams) from the **SCiMMA
Hopskotch** Kafka broker (`kafka.scimma.org`); a "ToO Producer" evaluates incoming
alerts against approved trigger criteria and forwards passing ones to the Engineering
Facility Database and scheduler. Super-K supernova alerts arrive via a GCN Kafka mirror.

For hazardous asteroids and interstellar objects there is **no automated path**: the
3I/ATLAS interstellar-comet ToO was triggered manually by committee. The SSSC NEOs
Working Group has defined filter criteria for NEO Rubin ToO triggers ("Filter Criteria
for near-Earth Object (NEO) Rubin ToO Triggers", v0.2), and FOMO already evaluates Scout
candidates against them — but nothing publishes the results anywhere Rubin can consume.
This bridge fills that gap. (Note: NEO trigger criteria still require Survey Cadence
Optimization Committee approval; until then the stream is advisory.)

## 2. The data source: JPL Scout API

- Endpoint: `https://ssd-api.jpl.nasa.gov/scout.api`
  ([docs](https://ssd-api.jpl.nasa.gov/doc/scout.html)), JSON with a
  `signature.version` (currently `1.3`) that must be validated.
- **Summary mode** returns the full current NEOCP candidate list including per-object
  `lastRun` timestamps — the natural change-detection key. **Object mode** (`tdes=`)
  returns per-object detail including sampled orbits.
- Fair use: one request at a time; no hard rate limits documented. A 10-minute poll
  cycle issuing one summary query plus sequential detail queries only for objects whose
  `lastRun` changed is comfortably within this.

## 3. Existing building blocks

| Piece | Where | What it provides |
|---|---|---|
| Scout API client + normalization | `tom_jpl/jpl.py` (`ScoutDataService`) | signature check; field mapping (`neoScore→neo_score`, arc hours→days, sexagesimal RA→deg) |
| Reconciliation loop | `tom_jpl` `ingest_scout` management command | cron-suitable full-list poll; upsert; departure sweep (with empty-response guard); MPC previous-designation enrichment |
| Change history | `tom_jpl/models.py` (`ScoutDetail`, `ScoutDetailHistory`) | one current row + append-only history unique on `(target, last_run)`; field-level diffing via `changes_from()`; `HISTORY_UNTRACKED_FIELDS` suppresses pure-ephemeris churn |
| Rubin ToO filter criteria | FOMO `solsys_code/rubin_too.py` | SSSC NEOs WG v0.2 §2.1 as pure predicates (`neoScore≥98`, `geocentricScore<2`, `rating≥3`, `rms<1.0`, `nObs>5` & `arc>1h`, `V>21.6/21.8` N/S, `unc_p1>60′/180′` N/S, `rate<25″/min`); §2.3 cancellation semantics |

Gaps to build: Kafka producer, container image, scheduler, deployment assets.
No existing Scout→Kafka producer was found anywhere in a search of GitHub and the web.

## 4. Broker options

### Option A — SCiMMA Hopskotch (recommended)

Publish with [`hop-client`](https://github.com/scimma/hop-client) to a topic on
`kafka.scimma.org` (e.g. `lco.scout-neo-too`, plus a `-test` topic for staging).

- This is **the broker Rubin's ToO Producer already subscribes to** — no Rubin-side
  infrastructure change is needed, only a subscription to the new topic.
- Free for scientific use; credentials via
  [my.hop.scimma.org/hopauth](https://my.hop.scimma.org/hopauth).
- Only the small poller container needs hosting; broker cost is zero.

### Option B — self-hosted Kafka on AWS (evaluated, not recommended)

- **AWS MSK Serverless**: ≈ $0.75/cluster-hour ⇒ **≈ $547/month minimum** before any
  traffic ([pricing](https://aws.amazon.com/msk/pricing/)) — disproportionate for a
  stream measured in messages per hour.
- **Single-node Kafka (KRaft) container** on EC2/Fargate: ≈ $30–60/month plus a public
  TLS + SASL endpoint, certificate management, patching, and single-node availability
  risk. Critically, **Rubin would need to add a bespoke consumer** pointed at our
  endpoint — contrary to their current all-through-Hopskotch model.

Option B is documented for completeness; Option A is recommended.

## 5. Service architecture

Three architectures were scoped:

| Option | Verdict |
|---|---|
| Django-free micro-poller (port the pure functions into a ~200 MB container, DynamoDB state) | Lightest, but **forks logic that is still evolving** (SSSC criteria v0.2; `tom_jpl` under active development) — rejected |
| **Standalone TOM Toolkit project depending on `tom_jpl`** | **Chosen.** Zero forking (ingest/reconciliation comes in as a pip dependency), Django admin over `ScoutDetail`/history for free, transactional outbox in Postgres, independent of the FOMO portal. ~1 GB image (tom_base deps) — acceptable for a k8s CronJob |
| Within FOMO itself | Couples Rubin alerting to a research portal that has never been deployed; `manage.py` system checks import FOMO's URLconf → `solsys_code.views` → `ephem_utils` → a ~1.6 GB SPICE-kernel download at import — rejected |

### Design (chosen architecture)

- **Django project shell** mirroring FOMO's config-only `src/fomo/` layout: settings +
  minimal urls (admin only; no `solsys_code`, hence no SPICE anywhere).
  `INSTALLED_APPS` = TOM Toolkit essentials + `tom_jpl` + a new `scout_publisher` app.
- **`scout_publisher` app** (starts in the bridge repo; designated follow-ups are
  upstreaming the publisher to `tom_jpl` as an optional extra and extracting the filters
  into a shared package used by both FOMO and the bridge):
  - `filters.py`: versioned copy of FOMO's `rubin_too.py` (with attribution);
  - `models.py`: `PublishedEvent` outbox — unique `(tdes, last_run, event_type)`,
    JSON payload, nullable `published_at`;
  - `publish_scout_events` management command: walk `ScoutDetail`/`ScoutDetailHistory`
    since the last watermark, derive events via `changes_from()` + `passes_filters()`,
    write outbox rows transactionally, publish unpublished rows via `hop-client`, and
    mark them published on broker ack. `--dry-run` prints events without publishing.
- **Poll cycle** (every 10 minutes):
  `manage.py ingest_scout --query-name=<broad-query> && manage.py publish_scout_events`.
  A bootstrap fixture provides the saved broad `DataServiceQuery` and service user that
  `ingest_scout` expects.
- **State**: Postgres. The transactional outbox gives exactly-once event emission: a
  failed publish leaves the watermark unadvanced, and the next cycle regenerates and
  retries; the unique idempotency key prevents duplicates.

## 6. Event model and message schema (proposal v1)

| Event type | Trigger |
|---|---|
| `new_candidate` | object newly passes **all** §2.1 filters (first time, or again after a `cancelled`/`left_neocp`) |
| `updated` | passing object has a new `lastRun` with tracked-field changes (ephemeris-only churn suppressed) |
| `cancelled` | previously-passing object now fails ≥1 filter (§2.3) while still on Scout |
| `left_neocp` | previously-passing object disappeared from the Scout list (designated / lost / impacted) |

Objects that never pass the filters generate no messages (state is still tracked so
`new_candidate` fires the moment one crosses the threshold).

```json
{
  "schema_version": "1.0",
  "event_type": "new_candidate",
  "event_id": "P12abcd:2026-07-15T10:31:00Z:new_candidate",
  "tdes": "P12abcd",
  "iau_designation": null,
  "scout": {
    "last_run": "2026-07-15 10:31:00", "neo_score": 100, "geocentric_score": 0,
    "impact_rating": 3, "rms": 0.4, "num_obs": 12, "arc_days": 0.31,
    "vmag": 22.1, "ra_deg": 187.3, "dec_deg": -12.4, "rate": 4.2,
    "uncertainty_p1_arcmin": 240.0, "ca_dist_ld": 0.8, "h_mag": 27.9,
    "url": "https://cneos.jpl.nasa.gov/scout/#/object/P12abcd"
  },
  "filters": {
    "version": "SSSC-NEO-WG-v0.2", "passes": true,
    "results": {"neo_score": true, "geocentric_score": true, "impact_rating": true,
                 "rms": true, "obs_arc": true, "vmag": true, "unc_p1": true, "rate": true}
  },
  "changes": {"num_obs": [8, 12], "rms": [0.9, 0.4]},
  "provenance": {"source": "JPL Scout API", "api_signature": "1.3",
                  "bridge_version": "0.1.0", "polled_at": "2026-07-15T10:40:12Z"}
}
```

Kafka message key = `tdes`; idempotency key = `(tdes, last_run, event_type)`.
Consumers must treat redelivery of the same `event_id` as a no-op. The schema is
deliberately provider-neutral so a future JPL-operated feed could be drop-in compatible.

## 7. Deployment: LCO GitOps / ArgoCD

LCO's Kubernetes clusters are cluster-api managed **on AWS** (`LCOGT/k8s-clusters`), so
"deploy to AWS" and "deploy via LCO's ArgoCD workflow" converge. Following LCO's
standard pattern:

- **App repo** `lsst-sssc/scout-alert-bridge`: the Django/TOM project, Dockerfile
  (`python:3.12-slim`; no SPICE), CI building a **public** image at
  `ghcr.io/lsst-sssc/scout-alert-bridge`.
- **Deploy repo** `LCOGT/scout-alert-bridge-deploy` from
  `LCOGT/deploy-repo-copier-template` (kpt + kustomize, staging/prod overlays): a
  **CronJob every 10 min** with `concurrencyPolicy: Forbid`, SCiMMA credentials as
  sealed-secrets, staging overlay pointed at the `-test` Hopskotch topic, registered as
  an ArgoCD Application.
- The cross-org app/deploy split matches LCO's existing pattern (deploy repos reference
  the app only as an image URL). Caveat: the ghcr package must be public, or the cluster
  needs an `imagePullSecret`.
- An AWS-native variant (EventBridge → Lambda container, ~$5–8/month, Terraform) was
  designed and remains an alternative **if the service should live outside LCO
  infrastructure** — but Lambda is not manageable by ArgoCD without Crossplane/ACK, so
  it sits outside LCO's GitOps workflow.

## 8. Failure modes and observability

- **Scout API down / empty response**: skip the cycle entirely (reusing `ingest_scout`'s
  empty-sweep guard) so no spurious `left_neocp` storm fires.
- **API signature ≠ 1.3**: hard stop before parsing; publish nothing; alert — schema
  drift needs human review.
- **Hopskotch unavailable**: outbox rows remain unpublished and the watermark does not
  advance; the next cycle retries. Idempotency keys prevent duplicates.
- **Dead-man alarm**: per-cycle heartbeat metric (Prometheus/Alertmanager per LCO
  cluster convention); optionally a low-frequency heartbeat message on a `*.heartbeat`
  topic so the Rubin side can also detect bridge death.
- Weekly digest of events/day and per-filter pass rates to validate trigger volumes
  against the SSSC document's expectations.

## 9. Costs

| Variant | Monthly cost |
|---|---|
| **Recommended**: SCiMMA broker + CronJob and small Postgres on an existing LCO cluster | ≈ $0 marginal (existing cluster capacity) |
| AWS-native standalone (Lambda + DynamoDB + Secrets Manager + CloudWatch) | ≈ $5–8 |
| Self-hosted single-node Kafka added | + $30–60 and ops labor |
| AWS MSK Serverless | ≈ $547 minimum — ruled out |

## 10. Open questions / coordination gates

1. **SCiMMA**: account and group provisioning, topic ACLs (write for the bridge, read
   for Rubin), retention/replay policy; institutional (`lco.*`) vs community (`scout.*`)
   topic naming.
2. **Rubin ToO team**: willingness of the ToO Producer to subscribe to a third-party
   topic; agreement on the §6 schema; SCOC approval of NEO trigger criteria.
3. **Ownership**: the ToO paper invites *JPL* to publish such a stream; this bridge is
   positioned as a community stopgap with a JPL-compatible schema. Confirm API fair-use
   with CNEOS for an institutional 10-minute poller.
4. **LCO infrastructure**: hosting cluster and namespace; Postgres provisioning; SCiMMA
   credential ownership; CronJob vs Deployment-with-loop convention.
5. **MPC designation enrichment**: whether `left_neocp` events need the IAU designation
   inline (reusing `ingest_scout`'s MPC previous-NEOCP lookup) or `tdes` suffices.

## 11. Prototype milestones (~4–5 engineering weeks; external coordination dominates)

- **M0** — circulate schema v1 to the Rubin ToO team and SSSC NEOs WG; request SCiMMA
  credentials and a dev topic (longest lead time — start first); put the §10.4 questions
  to LCO infrastructure.
- **M1** — create `lsst-sssc/scout-alert-bridge`: Django/TOM project shell + `tom_jpl` +
  `scout_publisher` app (filters copy, outbox model, `publish_scout_events --dry-run`);
  bootstrap fixture; Django-test-runner tests with canned Scout JSON fixtures covering a
  filter-crossing, an update, and a departure.
- **M2** — publish real events to the Hopskotch `-test` topic; verify with `hop subscribe`.
- **M3** — Dockerfile; local Postgres via docker compose; migration-job wiring; secrets
  handling.
- **M4** — `LCOGT/scout-alert-bridge-deploy` from the copier template; staging ArgoCD
  app; one-week soak on the dev topic; tune event-noise suppression.
- **M5** — production topic; Rubin subscribes; end-to-end latency measurement (Scout
  `lastRun` → Rubin receipt); joint review of a full candidate lifecycle; ownership
  handoff discussion.

## References

- Rubin ToO first-year operations paper: <https://arxiv.org/html/2607.00217v1>
- JPL Scout API documentation: <https://ssd-api.jpl.nasa.gov/doc/scout.html>
- SCiMMA hop-client tutorial:
  <https://github.com/scimma/hop-client/wiki/Tutorial:-using-hop-client-with-the-SCiMMA-Hopskotch-server>
- AWS MSK pricing: <https://aws.amazon.com/msk/pricing/>
- `tom_jpl`: <https://github.com/TOMToolkit/tom_jpl>
- SSSC NEOs WG, "Filter Criteria for near-Earth Object (NEO) Rubin ToO Triggers", v0.2
  (as implemented in `solsys_code/rubin_too.py`)
