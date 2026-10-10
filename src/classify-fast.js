/**
 * Routing decisions from a System One model, when the user has configured one.
 *
 * A System One model — TypeSafe's Jev is the first — takes structured state
 * and a set of typed questions and returns typed answers with a CALIBRATED
 * probability. It generates no text. That makes it a poor agent and a very
 * good classifier: `classify()` already asks exactly one question with a
 * closed set of answers, and already routes on the confidence that comes back.
 *
 * Why this is its own file rather than an entry in `providers.js`:
 * `PROVIDERS` is the registry of CHAT models. Everything that reads it —
 * onboarding, `listModels`, `isChatModel`, per-tier `modelFor()` — is choosing
 * a model to run an agent with. A System One model cannot run an agent: no
 * tool calling, no prose, so `doctor` would fail it and be right to. Putting
 * it in that table means offering it where it cannot work.
 *
 * Three rules hold this to the existing architecture:
 *
 *   1. It is OPT-IN. No `routing.classifier` block, no calls, no behaviour
 *      change. A second outbound destination for someone's task text and file
 *      list is not something to switch on for them.
 *   2. It NEVER decides enforcement. Guards, checkpoints, scope disjointness
 *      and build state stay deterministic. A guardrail that fires at p=0.87 is
 *      a guardrail nobody can trust.
 *   3. It ALWAYS falls back. Missing key, 401, timeout, a response shape we do
 *      not recognise — every one of them returns null and the existing model
 *      classifier runs. A beta API must not become the thing that decides
 *      whether this tool can route at all.
 *
 * Raw `fetch`, not the official SDK, because zero runtime dependencies is a
 * promise this project keeps. The SDK was installed once, read, and removed:
 * the request shapes below come from its source rather than from the prose
 * documentation, which had two fields wrong. `instructions`, not
 * `description`; `criteria` as a label→description map, not an `options`
 * array. Neither error could surface in testing, because a bad key fails
 * authentication before the body is validated — so every live check returned
 * 401 and looked like a key problem.
 *
 * If this drifts, `npm pack @typesafe-ai/sdk` and read `dist/index.mjs` (one
 * dependency-free file) rather than guessing from the docs again.
 */

import { redact } from './provider.js';
import { KeyRejected } from './providers.js';

/**
 * The three question primitives, in the shape the API actually takes.
 *
 * `criteria` being a map is the better shape for us anyway: each label carries
 * its own meaning, so an agent's own declared role describes choosing it.
 */
export const choice = (instructions, criteria) => ({ type: 'choice', instructions, criteria });
export const noul = (instructions, criteria = undefined) => ({ type: 'noul', instructions, criteria });
export const score = (instructions, criteria) => ({ type: 'score', instructions, criteria });

/**
 * The System One endpoints, kept as a table for the same reason `providers.js`
 * is one: so the wire address is never assembled from a guess at the call
 * site. One entry today; the shape is what matters.
 */
export const CLASSIFIERS = {
  typesafe: {
    label: 'TypeSafe (Jev)',
    base: 'https://api.typesafe.ai',
    path: '/v1/systemone',
    modelsPath: '/v1/models',
    keyEnv: 'TYPESAFE_API_KEY',
    defaultModel: 'jev-latest',
    signup: 'console.typesafe.ai/settings/keys',
    // What someone types. People say "jev" — the model — far more often than
    // "typesafe", the company, so both reach the same entry.
    aliases: ['jev', 'typesafe.ai', 'systemone', 'system-one'],
    // `jv_live_…`. Distinct from every chat provider's prefix, so unlike the
    // `sk-` the published docs show, this one can be recognised without any
    // chance of sending a router key to OpenAI.
    keyPattern: /^jv_/,
  },
};

/**
 * Is this a router key rather than a model key?
 *
 * Checked BEFORE `detectProvider`, because a router key must never be written
 * into a chat provider's variable — that is the key going to the wrong company
 * on the next request. A prefix is a convention rather than a contract, so a
 * key that matches nothing still reaches the right place by being named
 * (`jr-arch key jev <key>`).
 */
export function detectClassifier(key) {
  const k = String(key ?? '').trim();
  if (!k) return null;
  return Object.keys(CLASSIFIERS).find((id) => CLASSIFIERS[id].keyPattern?.test(k)) ?? null;
}

/** The registry id for whatever the user typed, or null. */
export function classifierByName(name) {
  const n = String(name ?? '').trim().toLowerCase();
  if (!n) return null;
  if (CLASSIFIERS[n]) return n;
  return Object.keys(CLASSIFIERS).find((id) => CLASSIFIERS[id].aliases?.includes(n)) ?? null;
}

/**
 * Jev advertises 70–500ms. A classifier that is slower than the model it is
 * meant to be faster than has no reason to exist, so the timeout is short and
 * expiring it costs nothing but the wait — the LLM path picks the task up.
 */
const TIMEOUT = 4000;
const MAX_FILES = 300;
const MAX_DUTIES = 4000;

/**
 * The configured classifier, or null.
 *
 * Null is the normal answer. It means the user never asked for one.
 */
export function classifierConfig(manifest) {
  const cfg = manifest?.classifier;
  if (!cfg?.provider) return null;
  const spec = CLASSIFIERS[cfg.provider];
  if (!spec) return null;
  return {
    provider: cfg.provider,
    label: spec.label,
    model: cfg.model || spec.defaultModel,
    keyEnv: cfg.keyEnv || spec.keyEnv,
    baseUrl: String(cfg.baseUrl || spec.base).replace(/\/$/, ''),
    path: spec.path,
    modelsPath: spec.modelsPath,
    signup: spec.signup,
  };
}

// ---------------------------------------------------------------------------
// The wire
// ---------------------------------------------------------------------------

/**
 * One request, with the key never leaving this function.
 *
 * `fetchImpl` is injectable so tests never touch the network, the same way
 * `listModels` does it.
 */
async function send({ config, key, path, method = 'POST', body = null, fetchImpl = fetch, timeout = TIMEOUT }) {
  if (!key) throw new Error(`$${config.keyEnv} is not set`);
  const control = new AbortController();
  const timer = setTimeout(() => control.abort(), timeout);
  let res;
  try {
    res = await fetchImpl(`${config.baseUrl}${path}`, {
      method,
      headers: {
        authorization: `Bearer ${key}`,
        ...(body ? { 'content-type': 'application/json' } : {}),
      },
      ...(body ? { body: JSON.stringify(body) } : {}),
      signal: control.signal,
    });
  } catch (err) {
    throw new Error(redact(err?.message ?? String(err), key));
  } finally {
    clearTimeout(timer);
  }
  return res;
}

/**
 * One System One request: structured state in, typed answers out.
 *
 * Throws on anything unusual. Every caller in this file catches and returns
 * null, because the whole point is that a failure here is not a failed run.
 */
export async function ask({ config, state, questions, fetchImpl = fetch, timeout = TIMEOUT } = {}) {
  const key = process.env[config.keyEnv] || '';
  const res = await send({
    config, key, path: config.path, fetchImpl, timeout,
    body: { model: config.model, state, questions },
  });
  if (!res.ok) {
    const text = redact(await res.text().catch(() => ''), key);
    throw new Error(`${config.label} answered ${res.status}: ${text.slice(0, 200)}`);
  }
  return res.json();
}

/**
 * One answer out of a response, without betting the run on the envelope.
 *
 * The published SDKs expose `.choice`, `.probabilities` and `.confidence`, but
 * the raw JSON these sit in is not something this repo can verify without a
 * key. So: look where an answer could reasonably be, accept the field names an
 * answer could reasonably have, and return null rather than guess. Null routes
 * the task to the model classifier, which is the correct outcome for a
 * response we do not understand — far better than pulling a tier name out of a
 * shape we assumed.
 */
export function answerFor(data, name) {
  const holder = data?.answers ?? data?.questions ?? data?.results ?? data;
  const a = holder?.[name];
  if (a === undefined || a === null) return null;
  // A bare scalar is a complete answer on its own: no confidence reported.
  if (typeof a === 'string' || typeof a === 'number' || typeof a === 'boolean') {
    return { value: a, confidence: null, probabilities: null };
  }
  const value = a.choice ?? a.value ?? a.answer ?? a.noul ?? a.score ?? null;
  if (value === null) return null;
  const confidence = num(a.confidence ?? a.probability ?? (typeof value === 'number' ? value : null));
  const probabilities = a.probabilities && typeof a.probabilities === 'object' ? a.probabilities : null;
  return { value, confidence, probabilities };
}

/**
 * A probability, or null for "none was reported".
 *
 * The null check is load-bearing: `Number(null)` is 0, which is a perfectly
 * finite number and a completely wrong answer. Read as 0 it means "no
 * confidence at all", so a response that simply did not mention confidence
 * would push every task up a tier through the floor bump.
 */
function num(v) {
  if (v === null || v === undefined || v === '') return null;
  const n = Number(v);
  return Number.isFinite(n) ? Math.min(1, Math.max(0, n)) : null;
}

// ---------------------------------------------------------------------------
// Proving a key
// ---------------------------------------------------------------------------

/**
 * Does this key work, and what does it reach?
 *
 * `GET /v1/models`, which is the same key check every chat provider gets and
 * the same reason: a wrong key is caught while setting up rather than as a 401
 * on the first task. An earlier version of this asked a throwaway inference
 * question instead, on the belief that a System One model published no model
 * list. It does. A GET costs nothing, spends no tokens, and returns the real
 * list rather than the one name we had hard-coded.
 *
 * Throws `KeyRejected` for a key the provider refused, so callers can tell
 * "wrong key" from "no network" and say something useful about each.
 */
export async function verifyKey({ provider = 'typesafe', key, baseUrl = null, fetchImpl = fetch, timeout = 10000 } = {}) {
  const spec = CLASSIFIERS[provider];
  if (!spec) throw new Error(`Unknown classifier provider "${provider}".`);
  const config = {
    provider,
    label: spec.label,
    model: spec.defaultModel,
    keyEnv: spec.keyEnv,
    baseUrl: String(baseUrl || spec.base).replace(/\/$/, ''),
    path: spec.path,
    modelsPath: spec.modelsPath,
    signup: spec.signup,
  };

  // The key being checked has not been saved yet, so it is passed in rather
  // than read from the environment — it must not be written before it works.
  const res = await send({ config, key, path: config.modelsPath, method: 'GET', fetchImpl, timeout });
  if (res.status === 401 || res.status === 403) {
    throw new KeyRejected(`${config.label} rejected that key.`);
  }
  if (!res.ok) {
    const text = redact(await res.text().catch(() => ''), key);
    throw new Error(`${config.label} answered ${res.status}: ${text.slice(0, 200)}`);
  }
  const models = modelNames(await res.json().catch(() => null));

  return {
    provider,
    label: config.label,
    keyEnv: config.keyEnv,
    models,
    // The configured default when the list does not contain it, so the caller
    // always has a name to show. A list that came back empty is not an error:
    // the key authenticated, which is what was being checked.
    model: models.includes(config.model) ? config.model : (models[0] ?? config.model),
  };
}

/**
 * Model names from the SDK's model list, without assuming the element shape.
 *
 * `models.list()` unwraps to the array itself; the raw `{ models: [...] }`
 * envelope is accepted too so this stays usable either way. Each entry may be
 * a bare string or an object naming itself, so both are read and anything else
 * is skipped rather than turned into "[object Object]" in a menu.
 */
export function modelNames(wire) {
  const list = Array.isArray(wire?.models) ? wire.models : Array.isArray(wire) ? wire : [];
  return list
    .map((m) => (typeof m === 'string' ? m : m?.id ?? m?.name ?? m?.model ?? null))
    .filter((n) => typeof n === 'string' && n.trim())
    .map((n) => n.trim());
}

// ---------------------------------------------------------------------------
// Tier selection
// ---------------------------------------------------------------------------

/**
 * Each agent as a labelled alternative: its name, described by what it says it
 * is for.
 *
 * `criteria` is a map rather than a list precisely so each option can explain
 * itself, and an agent already declares everything needed to do that — role,
 * scope, whether it repairs builds, whether it is terminal.
 */
export function criteriaFor(agents) {
  const out = {};
  for (const a of agents) {
    const bits = [
      a.role || null,
      a.owns?.length ? `owns ${a.owns.join(' ')}` : null,
      a.fixesBuild ? 'repairs broken builds' : null,
      a.terminal ? 'stops and asks a human rather than escalating' : null,
    ].filter(Boolean);
    out[a.name] = bits.join(' · ') || null;
  }
  return out;
}

const TIER_QUESTION =
  'Which agent should take this task? Choose by the state of the repository and ' +
  'the shape of the task, never by the language or framework the repository is ' +
  'written in — a CSS change in a Go repository is still front-end work.';

/**
 * The entry tier, or null to let the model classifier answer.
 *
 * Returns the same shape `classify()` returns, so `run.js` never learns that
 * there are two ways to get one.
 */
export async function fastClassify({
  task,
  buildGreen = null,
  files = [],
  agents = [],
  duties = '',
  manifest,
  config = classifierConfig(manifest),
  fetchImpl = fetch,
  onNotice = null,
} = {}) {
  if (!config || !agents.length) return null;
  const tiers = agents.map((a) => a.name);

  let data;
  try {
    data = await ask({
      config,
      fetchImpl,
      state: stateFor({ task, buildGreen, files, agents, duties }),
      // The labels come from the directory, never from a list here, so the rule
      // that no default agent name appears in executable code holds by
      // construction — and each label's description is the agent's OWN declared
      // role, which is what an agent describing itself is for.
      questions: { tier: choice(TIER_QUESTION, criteriaFor(agents)) },
    });
  } catch (err) {
    onNotice?.(`${config.label} classifier unavailable (${err.message}); using the model instead`);
    return null;
  }

  const answer = answerFor(data, 'tier');
  const tier = typeof answer?.value === 'string' ? answer.value.trim() : null;
  // An option outside the set we supplied is a response we do not understand.
  if (!tier || !tiers.includes(tier)) {
    onNotice?.(`${config.label} returned no usable tier; using the model instead`);
    return null;
  }

  return {
    tier,
    // No reported confidence is not the same as low confidence. Treating a
    // silent response as 0 would bump every task up a tier.
    confidence: answer.confidence ?? 1,
    reason: describe(answer, config),
    source: 'system-one',
    probabilities: answer.probabilities ?? null,
  };
}

/**
 * Why this tier — assembled from the distribution, because a System One model
 * writes no prose.
 *
 * The runner-up is the useful part: "junior-dev 0.71 · ui-editor 0.22" tells a
 * reader what the close call was, which a model's one-line rationale usually
 * does not. This string goes into the transcript and the handoff brief, so it
 * has to stand on its own there.
 */
function describe(answer, config) {
  const p = answer.probabilities;
  if (p && typeof p === 'object') {
    const ranked = Object.entries(p)
      .filter(([, v]) => Number.isFinite(Number(v)))
      .sort((a, b) => Number(b[1]) - Number(a[1]))
      .slice(0, 3)
      .map(([k, v]) => `${k} ${Number(v).toFixed(2)}`);
    if (ranked.length) return `${config.label}: ${ranked.join(' · ')}`;
  }
  const c = answer.confidence;
  return `${config.label} chose this tier${c === null ? '' : ` (calibrated ${c.toFixed(2)})`}`;
}

/**
 * The state a routing decision is made from.
 *
 * Structured, not a prompt. This is the difference between the two classifier
 * paths: the model needs DUTIES.md as prose to reason from, a System One model
 * takes the repository's actual state as fields. DUTIES.md still travels —
 * it is the user's own entry contract and it is their right to have it obeyed
 * — but as one field among the facts rather than as the whole instruction.
 */
export function stateFor({ task, buildGreen, files = [], agents = [], duties = '' }) {
  const listed = files.slice(0, MAX_FILES);
  return {
    task: String(task ?? ''),
    build: buildGreen === true ? 'green' : buildGreen === false ? 'red' : 'unknown',
    file_count: files.length,
    files: listed,
    agents: agents.map((a) => ({
      name: a.name,
      role: a.role || null,
      priority: a.priority,
      owns: a.owns ?? [],
      repairs_build: !!a.fixesBuild,
      terminal: !!a.terminal,
      escalates_to: a.escalatesTo ?? null,
    })),
    entry_rules: String(duties ?? '').slice(0, MAX_DUTIES),
  };
}

// ---------------------------------------------------------------------------
// Swarm selection
// ---------------------------------------------------------------------------

/**
 * Which agents a task actually touches.
 *
 * Not a choice — a subset — so it is one boolean question per agent, all in a
 * single request. `selectSwarm()` asks the same thing of a chat model and has
 * to parse an array out of free text, where a hallucinated name is possible;
 * here the questions are fixed and the answers are booleans.
 *
 * Keys are positional (`a0`, `a1`) rather than agent names: an agent may be
 * called anything a directory can be called, and the names are echoed back in
 * a response we would then have to match. Position cannot be misspelled.
 */
export async function fastSwarm({
  task,
  agents = [],
  manifest,
  config = classifierConfig(manifest),
  fetchImpl = fetch,
  floor = 0.5,
} = {}) {
  if (!config || !task || agents.length < 2) return null;

  const questions = {};
  agents.forEach((a, i) => {
    questions[`a${i}`] = noul(
      `Does this task genuinely require work inside \`${a.name}\`'s own scope` +
      `${a.owns?.length ? ` (${a.owns.join(', ')})` : ''}?`,
      {
        yes: `${a.name} has work to do here — it will run, and cost a model call.`,
        no: `${a.name} has nothing to do here.`,
      },
    );
  });

  let data;
  try {
    data = await ask({ config, fetchImpl, state: stateFor({ task, buildGreen: null, agents }), questions });
  } catch {
    // A failed selection is not a failed run: the caller has a heuristic.
    return null;
  }

  const picked = agents.filter((a, i) => {
    const answer = answerFor(data, `a${i}`);
    if (!answer) return false;
    const v = answer.value;
    if (typeof v === 'boolean') return v;
    const n = Number(v);
    return Number.isFinite(n) && n >= floor;
  });
  // One agent named is a real answer — it means this is not swarm work — but
  // zero means the questions were not understood, and that is a fallback.
  return picked.length ? picked.map((a) => a.name) : null;
}
