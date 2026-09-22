import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { agentDir } from './paths.js';
import { callModel, extractJson } from './provider.js';
import { readAgents, escalatesTo, buildFixer } from './agents.js';
import { fastClassify, fastSwarm } from './classify-fast.js';

/**
 * Tier selection. One model call returning strict JSON {tier, confidence,
 * reason}, per DUTIES.md.
 *
 * Entry tier comes from repo state and task shape, never from language or
 * framework — a CSS tweak in a Go repo is still ui-editor work — so the prompt
 * carries the user's own DUTIES.md entry rules rather than a copy of them
 * baked in here. DUTIES.md is the contract; this file only routes.
 */

/**
 * Where a low-confidence classification goes. Over-qualifying costs tokens;
 * under-qualifying costs a thrash loop, so the bump is always upward.
 *
 * Where a low-confidence classification goes is the AGENT's decision, read
 * from its `escalates_to`. An agent that declares itself terminal does not bump
 * at all, which is also what keeps a build-fixer from being routed out of a red
 * build — that would contradict DUTIES entry rule 1.
 */
// Nothing here knows the default agent names. A low-confidence result is
// bumped to whatever the classified agent declares as its successor, which is
// the same routing the ladder uses when an agent runs out of attempts.

const MAX_FILES = 300;

const SYSTEM = `You classify a coding task to exactly one agent tier, for the harness described below.

Reply with a single JSON object and nothing else — no prose, no code fence:
{"tier": "<tier name>", "confidence": <0.0-1.0>, "reason": "<one short sentence>"}

confidence is your own estimate that this tier is correct. Be honest: a low
number routes the task to a more senior tier, which is the cheaper mistake.

Choose the tier using the entry rules below. Classify by repo state and the
shape of the task, never by the language or framework the repo is written in.`;

export async function classify({
  task,
  buildGreen = null,
  files = [],
  manifest,
  duties,
  agents = [],
  dir = agentDir(),
  call = callModel,
  fast = fastClassify,
  onNotice = null,
} = {}) {
  const roster = agents.length ? agents : readAgents(dir);
  const tiers = roster.map((a) => a.name);
  if (!tiers.length) throw new Error('No agents installed to classify between.');
  const fallback = pickFallback(manifest, tiers);

  // 1. A pinned entry tier skips the call entirely. Zero network traffic.
  if (manifest.entry && manifest.entry !== 'auto') {
    if (!tiers.includes(manifest.entry)) {
      throw new Error(
        `routing.entry is "${manifest.entry}", which is not in the agents list (${tiers.join(', ')}).\n` +
        '  Fix it in .gitagent/agent.yaml, or set it to auto.',
      );
    }
    return { tier: manifest.entry, confidence: 1, reason: 'routing.entry pins the tier', source: 'config' };
  }

  // 2. DUTIES.md entry rule 1 is deterministic: a red build routes to
  //    build-doctor and nothing else runs until it is green. No point paying
  //    for a model call to rediscover that. Unknown (null) is not red.
  const fixer = buildFixer(roster);
  if (buildGreen === false && fixer) {
    return { tier: fixer.name, confidence: 1, reason: 'the build is red; nothing else runs until it is green', source: 'repo-state' };
  }

  const rules = duties ?? readDuties(dir);
  const floor = manifest.confidenceFloor ?? 0.6;

  // 3. A configured System One model answers this exact question — a closed
  //    set of options with a calibrated probability — in a fraction of the
  //    time and without generating a token of prose. It is opt-in and it is
  //    allowed to decline: null here means the user configured none, or the
  //    call failed, and the model below picks the task up either way.
  //
  //    Its confidence goes through the SAME floor as the model's, which is
  //    most of the reason to use it. The number a model reports about its own
  //    answer is a guess about a guess; a calibrated one makes the floor bump
  //    mean what `routing.classifier_confidence_floor` says it means.
  const quick = await fast({ task, buildGreen, files, agents: roster, duties: rules, manifest, onNotice });
  if (quick) return withFloor(quick, roster, fallback, floor);

  const response = await call(manifest, {
    system: `${SYSTEM}\n\nAvailable tiers, as they describe themselves:\n${describeAgents(roster)}\n\n--- DUTIES.md ---\n${rules}`,
    messages: [{ role: 'user', content: prompt(task, buildGreen, files) }],
    maxTokens: 256,
    temperature: 0,
  });

  const parsed = extractJson(response.text);
  const tier = typeof parsed?.tier === 'string' ? parsed.tier.trim() : null;

  // 3. A model that cannot return the shape does not get to pick the tier.
  //    That is what routing.degraded_fallback is for.
  if (!tier || !tiers.includes(tier)) {
    return {
      tier: fallback,
      confidence: 0,
      reason: tier
        ? `classifier returned unknown tier "${tier}"; using routing.degraded_fallback`
        : 'classifier did not return usable JSON; using routing.degraded_fallback',
      source: 'fallback',
    };
  }

  const confidence = clamp(parsed.confidence);
  const reason = typeof parsed.reason === 'string' ? parsed.reason.trim() : '';

  return withFloor({ tier, confidence, reason, source: 'model' }, roster, fallback, floor);
}

/**
 * Below the floor, route one tier UP.
 *
 * Shared by both classifier paths on purpose. The floor is a safety rule about
 * what to do with an uncertain answer, and it cannot depend on which kind of
 * model produced the uncertainty — a second copy of this is a second place for
 * the bump to quietly stop happening.
 */
function withFloor(result, roster, fallback, floor) {
  const { tier, confidence, reason } = result;
  if (confidence >= floor) return result;

  const bumped = bump(tier, roster, fallback);
  if (bumped === tier) return result;
  return {
    ...result,
    tier: bumped,
    reason: `${reason || 'no reason given'} (confidence ${confidence} below floor ${floor}; routed up from ${tier})`,
    source: 'floor-bump',
    classified: tier,
  };
}

/**
 * One step up, as the agent itself defines it.
 *
 * An agent with nowhere to escalate stays where it is: bumping it to a
 * "terminal" the harness picked would route work to an agent the user never
 * said should receive it.
 */
function bump(tier, roster, fallback) {
  const agent = roster.find((a) => a.name === tier);
  const next = agent ? escalatesTo(agent, roster) : fallback;
  return next ?? tier;
}

/**
 * Where a model that cannot classify sends the task.
 *
 * `routing.degraded_fallback` when the user named one and it is installed;
 * otherwise the LAST agent by priority, which is the most senior one present.
 * The previous version fell back to the literal name `senior-dev`, which meant
 * a repo whose agents are called something else degraded to a tier that does
 * not exist.
 */
function pickFallback(manifest, tiers) {
  const declared = manifest.degradedFallback;
  if (declared && tiers.includes(declared)) return declared;
  return tiers[tiers.length - 1];
}

function clamp(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return 0;
  return Math.min(1, Math.max(0, n));
}

function prompt(task, buildGreen, files) {
  const state = buildGreen === true ? 'green' : buildGreen === false ? 'red' : 'unknown (no verify command found)';
  const listed = files.slice(0, MAX_FILES);
  const more = files.length > listed.length ? `\n… and ${files.length - listed.length} more files` : '';
  return [
    `Task:\n${task}`,
    `\nBuild state: ${state}`,
    `\nRepository files:\n${listed.join('\n') || '(none listed)'}${more}`,
  ].join('\n');
}

/**
 * The entry rules, from the user's own DUTIES.md when there is one.
 *
 * A missing DUTIES.md is not an error. The loop already treats the file as
 * optional — it passes whatever is there and nothing when it is gone — and this
 * threw instead, so deleting a file the docs call optional broke every task
 * that needed classifying. What the classifier actually needs is the entry
 * rule, and the agents themselves declare enough to state it.
 */
function readDuties(dir) {
  const file = join(dir, 'DUTIES.md');
  if (existsSync(file)) return readFileSync(file, 'utf8');
  return [
    '(No DUTIES.md in this repository, so the default entry rules apply.)',
    '',
    '1. If the build is red, the agent that repairs builds takes it.',
    '2. Otherwise the task goes to the agent whose declared scope covers it,',
    '   lowest priority number first.',
    '3. A task no scope covers goes to the agent with no scope, or failing that,',
    '   to the last agent by priority.',
  ].join('\n');
}

/** What each agent says it is for, so the model can route without DUTIES.md. */
function describeAgents(agents) {
  return agents.map((a) => {
    const bits = [
      `priority ${a.priority}`,
      a.owns.length ? `owns ${a.owns.join(' ')}` : 'owns anything',
      a.fixesBuild ? 'repairs builds' : '',
      a.terminal ? 'terminal' : a.escalatesTo ? `escalates to ${a.escalatesTo}` : '',
    ].filter(Boolean).join(' · ');
    return `- ${a.name}: ${a.role || 'no role declared'} (${bits})`;
  }).join('\n');
}

/**
 * Which of these agents does the task actually span?
 *
 * `--swarm` used to fan out to every parallel agent that owned any file in the
 * repository, which is not a fact about the task at all: an agent owning
 * `**\/*.css` was "busy" in any repo containing CSS, and ran — and was billed
 * for — on a task about the database. The question is what the task touches, so
 * it is asked the same way the entry tier is.
 *
 * Returns null when the model cannot answer, and the caller falls back to the
 * ownership heuristic rather than refusing to swarm.
 */
const SWARM_SYSTEM = `You decide which agents a coding task needs.

Reply with a single JSON object and nothing else:
{"agents": ["name", ...]}

Name only agents whose own scope the task genuinely touches — they will run at
the same time, and each one costs the user a model call. One agent is a normal
answer. Never invent a name that is not listed.`;

export async function selectSwarm({ task, agents, manifest, call = callModel, fast = fastSwarm } = {}) {
  if (!task || !agents?.length) return null;

  // A subset question is one boolean per agent, which a System One model
  // answers in one request and cannot answer with a name that does not exist.
  const quick = await fast({ task, agents, manifest });
  if (quick) return quick;

  try {
    const response = await call(manifest, {
      system: `${SWARM_SYSTEM}\n\nAgents:\n${describeAgents(agents)}`,
      messages: [{ role: 'user', content: `Task:\n${task}` }],
      maxTokens: 200,
      temperature: 0,
    });
    const parsed = extractJson(response.text);
    const names = Array.isArray(parsed?.agents) ? parsed.agents.map((n) => String(n).trim()) : null;
    if (!names) return null;
    const known = names.filter((n) => agents.some((a) => a.name === n));
    return known.length ? known : null;
  } catch {
    // A failed selection is not a failed run: the caller has a heuristic.
    return null;
  }
}
