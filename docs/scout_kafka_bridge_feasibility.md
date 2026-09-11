# Feasibility study: a JPL Scout → Kafka bridge for Rubin ToO alerting

*Status: prototype running (first drafted 2026-07-17, updated 2026-09-10). Milestones
M1–M3 are complete and the bridge has been publishing to the Hopskotch topic
`Scout.scout-test` on a 10-minute cycle since 2026-08-31; see §12. The `tom_jpl` work
this design depends on was merged (PR #23, 2026-09-09) and released as `tom-jpl` 0.3.0
on PyPI on 2026-09-10, so the bridge no longer needs a git dependency. Remaining work
and the outstanding coordination gates are in §11–§12.*

## Summary

We assess the feasibility of a containerized service that polls the JPL/CNEOS **Scout**
NEOCP hazard-assessment system and republishes new, updated, and cancelled
Rubin-ToO-candidate objects as a **Kafka stream** consumable by the Vera C. Rubin
Observatory's Target-of-Opportunity system.

**Verdict: feasible, inexpensive, and well-timed.** Rubin's first-year ToO operations
paper ([arXiv:2607.00217](https://arxiv.org/html/2607.00217v1)) states that *"as of
August 24, 2026, no kafka stream for potentially hazardous asteroids is available"*, and
explicitly encourages *"the JPL-Scout program and the forthcoming JPL-NEO Surveyor
program to publish fully-machine-readable alerts to a Kafka stream, to support
localization efforts from Rubin ToO follow-up"*. Nearly all of the domain logic required
already exists in [`tom_jpl`](https://github.com/TOMToolkit/tom_jpl) (Scout ingestion and
change reconciliation) and in FOMO's `solsys_code/rubin_too.py` (the SSSC NEOs WG ToO
filter criteria). The remaining work is a Kafka publisher, a container, deployment
plumbing, and — as §7 establishes — a Scout filter class contributed to Rubin's ToO
Producer, which converts our alerts into the sky maps its scheduler actually consumes.

## 1. Background

Through its ToO program, Rubin reserves a fraction of survey time for rare,
time-sensitive events. The ToO alert-ingestion pipeline consumes external alerts
(currently LVK gravitational-wave and IceCube neutrino streams) from the **SCiMMA
Hopskotch** Kafka broker (`kafka.scimma.org`); a "ToO Producer" evaluates incoming
alerts against approved trigger criteria and republishes passing ones as HEALPix reward
maps into the Engineering Facility Database, which the Scheduler CSC polls. Super-K
supernova alerts arrive via a GCN Kafka mirror. §7 traces that pipeline through the
public repositories that implement it, which turns out to matter for scoping.

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
| Scout API client + normalization | `tom_jpl/jpl.py` (`ScoutDataService`) | signature check; field mapping (`neoScore→neo_score`, arc hours→days, sexagesimal RA→deg); optional query-level score/uncertainty/close-approach cuts |
| Ingestion | `rundataquery <query_id>` (tom_dataservices, via a saved broad `DataServiceQuery`) | cron-suitable full-list poll; upsert of targets + `ScoutDetail`/history rows. Caveat: catches its own failures and exits 0, so freshness must be monitored, not exit codes |
| Reconciliation loop | `tom_jpl` `updatescout` management command | single-request roster reconcile refreshing active candidates and retiring departures (`active=False`), with empty-response **and** partial-list guards; separate MPC Previous-NEOCP outcome pass recording `mpc_status` (designated/lost/dne/na/ns), `mpc_reference`, `merged_into`, and renaming the Target to its IAU designation. Two cadences: `--skip-designations` hourly, `--skip-reconcile` daily |
| Change history | `tom_jpl/models.py` (`ScoutDetail`, `ScoutDetailHistory`) | one current row + append-only history unique on `(target, last_run)`; field-level diffing via `changes_from()`; `HISTORY_UNTRACKED_FIELDS` suppresses pure-ephemeris churn (n.b. it includes `vmag` and `rate`, so filter evaluation must re-check each row, not rely on `changes_from()` alone) |
| Rubin ToO filter criteria | FOMO `solsys_code/rubin_too.py` | SSSC NEOs WG v0.2 §2.1 as pure predicates (`neoScore≥98`, `geocentricScore<2`, `rating≥3`, `rms<1.0`, `nObs>5` & `arc>1h`, `V>21.6/21.8` N/S, `unc_p1>60′/180′` N/S, `rate<25″/min`); §2.3 cancellation semantics |

Gaps to build: Kafka producer, container image, scheduler, deployment assets, and a
Scout filter class for Rubin's ToO Producer (§7).
No existing Scout→Kafka producer was found anywhere in a search of GitHub and the web.

## 4. Broker options

### Option A — SCiMMA Hopskotch (recommended)

Publish with [`hop-client`](https://github.com/scimma/hop-client) to a topic on
`kafka.scimma.org` (e.g. `lco.scout-neo-too`, plus a `-test` topic for staging).

- This is **the broker Rubin's ToO Producer already subscribes to**, so no new transport,
  broker relationship or Rubin-side *infrastructure* is needed. It does still require a
  Rubin-side **code** change — a Scout filter class in the producer, which converts the
  alert into the sky map the scheduler consumes — but that is a reviewable PR against an
  existing public repository. See §7; this is the single biggest correction to the
  original scoping, which assumed a subscription would suffice.
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
    since the last watermark, derive events by re-evaluating `passes_filters()` per row
    (with `changes_from()` supplying the tracked-field diff — filter-relevant `vmag`/`rate`
    are history-untracked, so pass/fail must not be inferred from diffs alone), write
    outbox rows transactionally, publish unpublished rows via `hop-client`, and mark
    them published on broker ack. `--dry-run` prints events without publishing.
- **Poll cycle** (every 10 minutes):
  `manage.py rundataquery <query_id> && manage.py updatescout --skip-designations &&
  manage.py publish_scout_events`, plus a daily `manage.py updatescout --skip-reconcile`
  for MPC outcomes (the Previous-NEOCP page holds months of departures; daily is kinder
  to the MPC). A bootstrap fixture provides the saved broad `DataServiceQuery` (no
  score/uncertainty cuts, so state tracking sees every candidate) and service user that
  `rundataquery` expects.
- **State**: Postgres. The transactional outbox gives exactly-once event emission: a
  failed publish leaves the watermark unadvanced, and the next cycle regenerates and
  retries; the unique idempotency key prevents duplicates.

## 6. Event model and message schema (proposal v1)

| Event type | Trigger |
|---|---|
| `new_candidate` | object newly passes **all** §2.1 filters (first time, or again after a `cancelled`/`left_neocp`) |
| `updated` | passing object has a new `lastRun` with tracked-field changes (ephemeris-only churn suppressed) |
| `cancelled` | previously-passing object now fails ≥1 filter (§2.3) while still on Scout |
| `left_neocp` | previously-passing object disappeared from the Scout list (designated / lost / impacted). Enriched once `updatescout`'s MPC pass settles the outcome: `mpc_status` (designated/lost/dne/na/ns), `mpc_reference` (e.g. an MPEC), `merged_into`, and the IAU designation the Target was renamed to |

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
Field names track `tom_jpl` 0.3.0's `ScoutDetail` where they exist (`arc` is stored in
days, `ca_dist` in lunar distances, `uncertainty_p1` in arcmin; units are suffixed here
for self-description). `h_mag` is not stored by `tom_jpl` and would come from the Scout
object-mode response at publish time — or be dropped from v1 if not worth the extra query.
Consumers must treat redelivery of the same `event_id` as a no-op. The schema is
deliberately provider-neutral so a future JPL-operated feed could be drop-in compatible.

## 7. Rubin-side integration: what actually consumes the stream

Publishing to Hopskotch is necessary but not sufficient. Rubin's ToO Producer does not
forward alert payloads — it **converts** them into a HEALPix reward map. The deployed
pipeline is entirely public, and traces as follows (note the receiver lives in the SCiMMA
organization, not an `lsst` one):

| Stage | Where | What it does |
|---|---|---|
| Kafka receiver | [`scimma/rubin-ToO-producer`](https://github.com/scimma/rubin-ToO-producer) — `forward_alerts.py`, image `lsstts/rubin_too_producer` | subscribes to Hopskotch via `hop`; per-source `AlertFilter` subclasses decide follow-up and build the sky map |
| Deployment + config | [`lsst-sqre/phalanx`](https://github.com/lsst-sqre/phalanx) `applications/rubin-too-producer` | input topic, a `filters:` topic→filter map, output URL, SCiMMA credentials from Vault |
| EFD topics | phalanx `applications/sasquatch/charts/scimma` | declares Kafka topics `lsst.scimma.too.alert` and `lsst.scimma.too.alert.test` |
| Scheduler consumer | [`lsst-ts/ts_scheduler`](https://github.com/lsst-ts/ts_scheduler) `python/lsst/ts/scheduler/too_client.py` | `TooClient` polls the **EFD** (InfluxDB), not Kafka; configured in `ts_config_scheduler` (`topic_name: lsst.scimma.too_alert`, 8-day lookback) |
| Scheduling | [`lsst/rubin_scheduler`](https://github.com/lsst/rubin_scheduler) `scheduler/utils/too_objects.py`, `surveys/too_scripted_surveys.py` | `TargetoO` objects consumed by the feature-based scheduler |

The producer's output schema (`output_schema.json`, Avro record `lsst.scimma.too_alert`)
is narrow: `source`, `instrument[]`, `alert_type`, `event_trigger_timestamp`,
`reward_map` (boolean HEALPix array, nested ordering), `reward_map_nside`, `is_test`,
`is_update`, `timestamp`. **Everything else in the §6 payload is dropped on the Rubin
path** — that schema serves other subscribers and the scientific record; Rubin needs a
sky map.

### Consequences for this project

- **A new filter class is required**, contributed as a PR to `scimma/rubin-ToO-producer`:
  a `ScoutAlertFilter` registered in `filter_constructors` alongside `lvk_gw`,
  `icecube_nu` and `superk_sn`, implementing `is_test`, `alert_identifier`,
  `overrides_previous`, `should_follow_up` and `generate_scheduling_data`. It must turn
  our RA/Dec plus positional uncertainty into a binary HEALPix map — the same geometry
  the SSSC §2.1 `unc_p1` criterion already trades on. A `filters:` entry and the topic go
  into Phalanx alongside it. This is a reviewable code contribution to existing public
  Rubin/SCiMMA repositories, not new infrastructure, but it is more than a subscription.
- **`alert_type` is a categorization, not a lifecycle code**: the LVK filter emits
  `GW_case_B`, `GW_case_D`, `lensed_BNS_case_A`, `BBH_case_A` and so on, one per approved
  trigger case. Ours should follow the same convention (e.g. `NEO_case_*`), which makes
  the SCOC approval in §11.2 a naming gate as well as a scientific one.
- **Mark test traffic with the Hopskotch `_test` message header.** The base
  `AlertFilter.is_test` checks that transport header, so setting it is more useful to
  Rubin than a separate `-test` topic alone — and `TooClient` independently skips
  `is_test` alerts on the scheduler side.
- **Rubin already operates a SCiMMA group, `rubin-too-dev`** (its current input topic is
  `rubin-too-dev.lvk-test-alerts`), a concrete starting point for the §11.1 naming and
  ACL conversation. Note the deployed config takes a *single* input topic, so adding
  Scout may need multi-topic support or a second producer deployment.
- **No cancellation path exists downstream** — see §11.5.

## 8. Deployment: LCO GitOps / ArgoCD

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

## 9. Failure modes and observability

- **Scout API down / empty or truncated response**: skip the cycle entirely (reusing
  `updatescout`'s empty-response and partial-list guards) so no spurious `left_neocp`
  storm fires. Because `rundataquery` exits 0 even on failure, the heartbeat must be
  a data-freshness check (e.g. newest `last_run` ingested), not a process exit code.
- **API signature ≠ 1.3**: hard stop before parsing; publish nothing; alert — schema
  drift needs human review.
- **Hopskotch unavailable**: outbox rows remain unpublished and the watermark does not
  advance; the next cycle retries. Idempotency keys prevent duplicates.
- **Dead-man alarm**: per-cycle heartbeat metric (Prometheus/Alertmanager per LCO
  cluster convention); optionally a low-frequency heartbeat message on a `*.heartbeat`
  topic so the Rubin side can also detect bridge death.
- Weekly digest of events/day and per-filter pass rates to validate trigger volumes
  against the SSSC document's expectations.

## 10. Costs

| Variant | Monthly cost |
|---|---|
| **Recommended**: SCiMMA broker + CronJob and small Postgres on an existing LCO cluster | ≈ $0 marginal (existing cluster capacity) |
| AWS-native standalone (Lambda + DynamoDB + Secrets Manager + CloudWatch) | ≈ $5–8 |
| Self-hosted single-node Kafka added | + $30–60 and ops labor |
| AWS MSK Serverless | ≈ $547 minimum — ruled out |

## 11. Open questions / coordination gates

1. **SCiMMA**: account and group provisioning, topic ACLs (write for the bridge, read
   for Rubin), retention/replay policy; institutional (`lco.*`) vs community (`scout.*`)
   topic naming.
2. **Rubin ToO team / SCiMMA**: willingness of the ToO Producer to subscribe to a
   third-party topic; agreement on the §6 schema; SCOC approval of NEO trigger criteria
   and the resulting `alert_type` case names. Now also, per §7: who writes and reviews
   the `ScoutAlertFilter` in `scimma/rubin-ToO-producer` (us, offered as a PR, seems
   likeliest), what sky-map convention it should use, whether a second producer
   deployment or multi-topic input is preferred, and whether the unreachable retraction
   handling noted in §11.5 is intentional.
3. **Ownership**: the ToO paper invites *JPL* to publish such a stream; this bridge is
   positioned as a community stopgap with a JPL-compatible schema. Confirm API fair-use
   with CNEOS for an institutional 10-minute poller.
4. **LCO infrastructure**: hosting cluster and namespace; Postgres provisioning; SCiMMA
   credential ownership; CronJob vs Deployment-with-loop convention.
5. **MPC outcome enrichment — resolved.** `tom_jpl` 0.3.0 settles departures via
   `updatescout --skip-reconcile` (`mpc_status`, `mpc_reference`, `merged_into`, and a
   rename to the IAU designation), but a day after the fact: reconciliation runs every
   cycle and the MPC pass daily, so in practice most `left_neocp` events publish with
   `mpc_status` still null.

   Precedent says publish anyway, and do not follow up on the actionable topic. LVK's
   `RETRACTION` — the closest analogue, and one the Rubin ToO Producer already consumes —
   "provide[s] only the name": `event` and `external_coinc` are null and there is no
   machine-readable reason of any kind. GCN's multi-mission core `Alert.schema.json`,
   shared by the Super-K and IceCube notices, likewise types only the transition
   (`initial|subsequent|update|retraction`) and carries no reason beyond an opt-in
   free-text `additional_info` comment. Where the "why" does travel, it travels on a
   *separate* channel: GCN streams structured Notices and human-readable Circulars as
   different Kafka topics. The Rubin ToO paper itself never mentions retraction handling.

   Rubin's own implementation confirms it from the other side. In
   `scimma/rubin-ToO-producer` every filter defines an `overrides_previous` carrying a
   retraction branch, but `process()` consults `should_follow_up()` first and returns
   early when it fails — and each filter's `should_follow_up` admits only a specific
   non-retraction `alert_type` (`INITIAL` for LVK, `initial`/`update` for IceCube). A
   retraction therefore never reaches the override logic, and the output schema (§7) has
   no field able to express one. Our `cancelled` and `left_neocp` events consequently
   have nowhere to land on the Rubin path today. Whether that is deliberate — an exposure
   already taken cannot be un-taken — or a latent bug is worth putting to the ToO team
   as part of §11.2.

   **Decision**: `left_neocp` publishes immediately, unblocked and terminal, carrying the
   outcome fields only when they happen to be settled already; no second event on the ToO
   topic. The outcome stays available in the bridge database and Django admin, and is
   derivable by any consumer from the public MPC page.

   **Open only if a subscriber asks for the outcome to be pushed**: the precedent-aligned
   answer is a separate informational topic, mirroring GCN's Notices/Circulars split,
   rather than a follow-up on the ToO topic. Were it ever added to the main topic it
   would need its own `event_type` (LVK never sends anything after a retraction) and
   `in_candidate_set = False`, or a "last event wins" consumer could resurrect a retired
   object. Note that publishing a structured `mpc_status` enum at all is *ahead* of
   precedent rather than merely different — no existing consumer will expect it, which is
   a further argument for keeping it ignorable.

## 12. Prototype milestones (~5–6 engineering weeks; external coordination dominates)

*Status as of 2026-09-10: M1–M3 complete, M0 partly resolved, M4 next. `tom-jpl` 0.3.0
is on PyPI (requires `tomtoolkit>=3.0.1`), closing the "unreleased dependency" caveat
under which M1–M3 were built: the bridge's git pin on the PR branch can be replaced by
`tom-jpl>=0.3.0` before M4 containerises it.*

- **M0 — partly done.** SCiMMA side resolved 2026-08-24: we are Owner of the `Scout`
  hopauth group, so topic creation and write credentials turned out to be self-serve and
  needed no SCiMMA action. `Scout.scout-test` and `Scout.scout-prod` exist; `rubin-too-dev`
  holds Read on `-test` as of 2026-08-31. The §11.2 conversation opened with the Rubin ToO
  team on 2026-09-01 (outcomes still to be recorded). *Outstanding*: schema v1 circulated
  to the SSSC NEOs WG, the §11.4 LCO infrastructure questions, and confirming which group
  Rubin's production ToO Producer authenticates as before granting on `scout-prod`.
- **M1 — done 2026-07-18.** `lsst-sssc/scout-alert-bridge`: Django/TOM project shell +
  `tom_jpl` + `scout_publisher` app (filters copy, outbox model, `publish_scout_events`);
  bootstrap fixture; Django-test-runner tests with canned Scout JSON fixtures covering a
  filter-crossing, an update, and a departure.
- **M2 — done 2026-08-31.** 25 events published to `Scout.scout-test` and independently
  verified off the broker with `hop subscribe`. Run under `--relaxed-filters`, which gates
  on the identity filters only and stamps `provenance.filter_mode='relaxed_test'`: a real
  `impact_rating >= 3` object is rare enough that this is the only practical way to
  exercise the path end to end. Payloads still report the full, honest filter results.
- **M3 — done 2026-07-18.** Dockerfile; local Postgres via docker compose; migration
  wiring; secrets handling. Scheduling is currently a host `cron` entry guarded by
  `flock`; the containerised CronJob arrives with M4.
- **M4 — next.** `LCOGT/scout-alert-bridge-deploy` from the copier template; staging ArgoCD
  app; two CronJob manifests (the 10-minute cycle and the daily MPC pass) with
  `concurrencyPolicy: Forbid` replacing the host cron and `flock`; one-week soak on the dev
  topic; tune event-noise suppression.
- **M5** — `ScoutAlertFilter` for `scimma/rubin-ToO-producer` (§7): sky-map generation
  from RA/Dec and positional uncertainty, `alert_type` case names, `_test` header
  handling, unit tests against canned bridge messages; offered as a PR, with the
  matching Phalanx `filters:`/topic entry. Depends on M0 agreement and M2 sample traffic,
  and carries its own external review cycle.
- **M6** — production topic; Rubin subscribes and the filter is deployed; end-to-end
  latency measurement (Scout `lastRun` → Rubin receipt); joint review of a full candidate
  lifecycle; ownership handoff discussion.

## References

- Rubin ToO first-year operations paper: <https://arxiv.org/html/2607.00217v1>
- JPL Scout API documentation: <https://ssd-api.jpl.nasa.gov/doc/scout.html>
- SCiMMA hop-client tutorial:
  <https://github.com/scimma/hop-client/wiki/Tutorial:-using-hop-client-with-the-SCiMMA-Hopskotch-server>
- IGWN/LVK Public Alerts User Guide, "Alert Contents" (retraction semantics, §11.5):
  <https://emfollow.docs.ligo.org/userguide/content.html>
- GCN unified multi-mission schema (core `Alert.schema.json`, `AdditionalInfo`):
  <https://gcn.nasa.gov/docs/notices/schema> and <https://github.com/nasa-gcn/gcn-schema>
- AWS MSK pricing: <https://aws.amazon.com/msk/pricing/>
- `tom_jpl`: <https://github.com/TOMToolkit/tom_jpl>
- Rubin ToO Producer (Kafka receiver and per-source alert filters):
  <https://github.com/scimma/rubin-ToO-producer>
- Its deployment and configuration (`applications/rubin-too-producer`, and the
  `sasquatch/charts/scimma` EFD topics): <https://github.com/lsst-sqre/phalanx>
- Scheduler-side consumer `TooClient`: <https://github.com/lsst-ts/ts_scheduler> and its
  configuration <https://github.com/lsst-ts/ts_config_scheduler>
- Feature-based scheduler ToO objects and surveys:
  <https://github.com/lsst/rubin_scheduler>
- SSSC NEOs WG, "Filter Criteria for near-Earth Object (NEO) Rubin ToO Triggers", v0.2
  (as implemented in `solsys_code/rubin_too.py`)
