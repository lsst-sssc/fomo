Run Identity and Unattended Invocation Spike
============================================

This document records the investigation spike that settled how a routine, non-campaign
observation from FOMO's LCO and Gemini queue sync commands gets a persistent
``CampaignRun`` identity, and how FOMO's three sync commands can run unattended on a real
schedule. It was written after a live investigation (2026-09-01 through 2026-09-02) that
read the real dev database, applied throwaway schema changes to a disposable copy of it,
inspected a real operator-supplied classical schedule file, and probed the real interim
host and a stand-in container image, rather than reasoning from documentation alone. No
``CampaignRun`` schema migration, no adapter code, and no scheduler entry point was built
during this spike — the deliverable is this durable summary and its full-detail
companion, ``31-DECISION.md`` (originally at
``.planning/phases/31-foundation-spikes-run-identity-unattended-invocation/31-DECISION.md``;
this project's milestone-archival workflow moves completed phase directories to
``.planning/phases-archive/`` once their milestone closes, so check there first if the
original path no longer resolves).

Background
----------

Today, ``CampaignRun.campaign`` is a required foreign key to a coordinated campaign
``TargetList`` — every row needs one. But a routine LCO or Gemini queue observation, or a
classically scheduled night, is not part of any coordinated campaign. Before v2.3's
adapters (Phase 32) can write a ``CampaignRun`` for one of those, four concrete questions
had to be settled against real data, so no later phase has to re-derive the answer from
scratch, or discover it the hard way once adapters are already writing:

* Can a ``CampaignRun`` exist without a campaign at all, and if so, in what shape — a
  nullable foreign key, a shared placeholder campaign, or one placeholder per proposal?
* Once a non-campaign run can exist, what field lets FOMO find that same run again on a
  second sync pass, without colliding with the identity rules two campaign runs already
  rely on?
* Is the classical adapter's existing five-minute telescope/instrument/start-time match
  good enough on its own to tell two different proposals' runs apart, or does it need a
  proposal-specific key?
* How does FOMO run its three sync commands with nobody typing a command, on the real
  machine this project runs on today and the AWS cluster it will run on eventually?

Every later phase in this milestone executes these decisions instead of making them:
Phase 32's adapters write against whatever schema shape and identity field this spike
recommends, and Phase 34's scheduler entry point is built directly against the invocation
mechanism this spike verifies.

Decisions
---------

**Schema shape and write-time identity (SCHEMA-01/02/03)**

.. list-table::
   :header-rows: 1
   :widths: 22 50 12

   * - Topic
     - Decision
     - Phase
   * - Schema shape for a non-campaign run
     - Make ``CampaignRun.campaign`` nullable (``null=True, blank=True``), chosen over a
       shared sentinel campaign and a per-proposal placeholder campaign. A shared
       sentinel collides: four real telescope/instrument/window combinations already
       recur across different real campaigns today, so collapsing every non-campaign run
       onto one shared campaign value would refuse a second row for at least those four
       combinations. A per-proposal placeholder only splits that collision risk by
       proposal, it does not eliminate it, and it opens an unanswered question of its own
       — what happens to a placeholder's rows once its proposal later gets a real
       campaign. Nullable costs a single field-level migration and needs no backfill: 0 of
       the 49 real ``CampaignRun`` rows today have a null campaign, so the change only
       ever affects rows a future adapter writes.
     - 32
   * - Write-time identity field and constraint
     - A new ``source_identifier`` field (nullable ``CharField``) with its own partial
       unique constraint, additive alongside both of ``CampaignRun``'s existing partial
       constraints — proven, not just argued: after adding the field to a disposable copy
       of the database, both existing constraints still refused a genuine duplicate
       exactly as before. ``source_identifier`` shares no field with either existing
       constraint's field set, so satisfying one can never force a violation of the
       other.
     - 32
   * - Per-ingest-path value
     - ``sync_lco_observation_calendar`` writes the real LCO portal request URL it
       already extracts today; ``sync_gemini_observation_calendar`` writes its own
       constructed ``GEM:{program}/{observation-id}`` key; ``load_telescope_runs`` writes
       a synthesized ``CLASSICAL:{telescope}:{instrument}:{start-time}`` key, matching
       the same three fields its existing tolerance match already uses. All three
       round-tripped cleanly through a find-or-create keyed on this field alone, with no
       duplicate row created on a second write.
     - 32
   * - Classical adapter's tolerance match, on its own
     - Not sufficient. The existing five-minute telescope/instrument/start-time
       tolerance match (and its ``source_identifier`` mirror) cannot tell apart two
       genuinely different proposals allocated the same telescope, the same instrument
       and the same night — neither the run's status nor any proposal identifier is part
       of that lookup, so the second proposal's write silently overwrites the first's
       record rather than creating a second one. A real classical schedule file sample
       confirms a proposal code is not a reliable stand-in today: only one of three real
       sample lines carried one at all, and where present it did not even parse under the
       loader's current line grammar. The gap is accepted as a documented, low-frequency
       risk for now — kept as today's default because it correctly handles the great
       majority of real lines — with two concrete follow-on items named for whichever
       phase implements it: teach the parser to recognise a leading proposal-code token,
       then decide whether to fold it into the identity key once one is reliably
       extractable.
     - 32

**Unattended invocation (SCHED-07)**

**Correction, recorded 2026-09-02, from this phase's own source-code reading — read this
before the table below, which it qualifies.** An earlier planning document (this phase's
own context document) described FOMO's existing staff-notification helper as
"``mail_admins()``-based." It is not: a full-codebase search found zero uses of Django's
``mail_admins()`` anywhere in this project. The helper actually in use,
``campaign_views.py``'s ``_notify_staff()``, emails every staff user with a non-blank
email on file directly via ``send_mail()``. The table below is written against the
corrected mechanism.

.. list-table::
   :header-rows: 1
   :widths: 22 50 12

   * - Topic
     - Decision
     - Phase
   * - Invocation mechanism
     - Cron plus a per-command file lock (``flock``), running inside the FOMO container —
       the same mechanism on today's interim host and on the eventual AWS Kubernetes
       deployment, so migrating between them needs no scheduling redesign. Confirmed
       against the real interim host: the lock utility is already installed there, and
       the real crontab already runs this project's own management commands today. The
       **container image and the AWS deployment target are both unconfirmed** — this
       repository tracks no container build file at all, so the checks that could
       confirm either scope were run against a generic stand-in image instead, which
       proves nothing about an image that does not exist yet. Whoever writes FOMO's own
       container definition inherits the requirement to install both cron and the lock
       utility inside it.
     - 34
   * - Overlap prevention
     - ``flock -n`` (non-blocking) against one lock file per management command name,
       never one shared lock across every scheduled command. A tick that arrives while
       the previous run still holds the lock is skipped outright, not queued and not
       blocked — chosen because the real crontab on the interim host today runs three
       FOMO management commands with zero overlap guards between them, a live condition
       this spike measured directly rather than assumed.
     - 34
   * - Credential handling
     - Environment variables, extending this project's existing ``FINK_CREDENTIAL_*``
       naming convention for any new LCO/Gemini credential a scheduled command needs.
       Never a command-line argument to the management command and never embedded in the
       cron line itself — both are readable by any local process listing or, depending on
       file permissions, other local users of the same host.
     - 34
   * - Missed-invocation visibility
     - Two independent layers, because neither alone can catch every failure mode: an
       in-command extension of this project's existing staff-notification email, and an
       external heartbeat dead-man's-switch that fires when an expected ping fails to
       arrive. Only the external layer can catch the scheduler itself never invoking the
       command at all. The heartbeat's technical reachability from the interim host is
       confirmed (a real outbound request to a heartbeat service succeeded); whether
       pinging a third-party service is acceptable on policy or compliance grounds from
       the eventual AWS deployment **remains unresolved** and is a question for whoever
       owns that deployment's network policy, not something a development-host probe can
       answer.
     - 34

Future scope
------------

See ``31-DECISION.md`` (path note above) for the full evidence each of these decisions
rests on — including the real dev-database snapshot and constraint-probe output, the
grep-derived inventory of every site that reads ``CampaignRun.campaign`` and would need a
null guard under the chosen schema shape, the real classical schedule-file sample and its
parse results, and the verbatim host/container probe transcripts. These are
recommendations for Phase 32 and Phase 34 to implement — none of it is implemented in
this spike.

Explicitly still open, carried forward rather than answered here:

* Whether a classical run's proposal code should be folded into its identity key once the
  parser can recognise one, versus keeping the current tolerance-match-only key and
  accepting the documented two-proposals-same-night gap as a permanent risk (Phase 32).
* Whether ``source_identifier`` should be promoted to the primary lookup key for
  adapter-written rows once Phase 32's adapters exist, rather than added alongside the
  existing campaign-plus-window lookup (Phase 32).
* What happens to a ``CampaignRun`` row's identity once its proposal later acquires a
  real coordinated campaign — settled for the chosen nullable-FK shape (the row's
  campaign field is simply updated from null to the real value; nothing is re-pointed or
  duplicated), but the same question was never fully answered for the two rejected
  candidate shapes (Phase 32).
* Whether the FOMO container image, whenever one is written, actually ships the lock
  utility and an HTTP client — a stand-in image checked during this spike showed the lock
  utility present but the client absent, proving nothing about FOMO's own image but
  demonstrating exactly the kind of gap a slim base image can hide (Phase 34).
* Whether pinging a third-party heartbeat service from the eventual AWS deployment is
  acceptable on policy or compliance grounds — confirmed only as technically reachable
  from today's interim host (Phase 34).
* Whether several scheduled commands due at the same minute need a specified relative
  invocation order — depends on a command set this milestone has not finalised yet
  (Phase 34).
