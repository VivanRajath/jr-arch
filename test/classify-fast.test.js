import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, mkdtempSync, cpSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import {
  CLASSIFIERS, classifierConfig, fastClassify, fastSwarm, answerFor, stateFor,
  classifierByName, verifyKey, detectClassifier,
} from '../src/classify-fast.js';
import { KeyRejected, detectProvider } from '../src/providers.js';
import { envTemplate } from '../src/env.js';
import { classify, selectSwarm } from '../src/classify.js';
import { readManifest, keyEnvs, setClassifier } from '../src/config.js';
import { readAgents } from '../src/agents.js';
import { TEMPLATES } from '../src/paths.js';

/**
 * The System One classifier path.
 *
 * The invariant every test here defends: this is an OPTIONAL accelerator that
 * is allowed to fail. Not configured, no key, a 500, a timeout, a shape nobody
 * recognises — all of them land on the model classifier rather than on the
 * user.
 */

const DUTIES = readFileSync(join(TEMPLATES, 'DUTIES.md'), 'utf8');
const BASE = readManifest(join(TEMPLATES, 'agent.yaml'));
const AGENTS = readAgents(TEMPLATES);

const CONFIGURED = {
  ...BASE,
  classifier: { provider: 'typesafe', model: null, keyEnv: 'TYPESAFE_TEST_KEY', baseUrl: null },
};

/** A stub System One endpoint. Records requests, replies with whatever is given. */
function endpoint(reply, { status = 200 } = {}) {
  const calls = [];
  const fn = async (url, init) => {
    calls.push({ url, init, body: JSON.parse(init.body) });
    const body = typeof reply === 'function' ? reply(JSON.parse(init.body)) : reply;
    return {
      ok: status >= 200 && status < 300,
      status,
      json: async () => body,
      text: async () => JSON.stringify(body),
    };
  };
  fn.calls = calls;
  return fn;
}

/** A stub chat model, so a fallback is visible as a call that happened. */
function stub(text) {
  const calls = [];
  const fn = async (m, req) => {
    calls.push(req);
    return { text, toolCalls: [], stopReason: 'end_turn', raw: {} };
  };
  fn.calls = calls;
  return fn;
}

const withKey = async (value, fn) => {
  const had = Object.prototype.hasOwnProperty.call(process.env, 'TYPESAFE_TEST_KEY');
  const prev = process.env.TYPESAFE_TEST_KEY;
  if (value === null) delete process.env.TYPESAFE_TEST_KEY;
  else process.env.TYPESAFE_TEST_KEY = value;
  try { return await fn(); } finally {
    if (had) process.env.TYPESAFE_TEST_KEY = prev;
    else delete process.env.TYPESAFE_TEST_KEY;
  }
};

const answered = (tier, probabilities = null, confidence = 0.9) =>
  ({ answers: { tier: { choice: tier, confidence, probabilities } } });

describe('configuration', () => {
  // The default scaffold must not send anything anywhere new.
  test('the shipped template configures no classifier', () => {
    assert.equal(BASE.classifier, null);
    assert.equal(classifierConfig(BASE), null);
  });

  test('a configured block resolves through the registry, never a guess', () => {
    const cfg = classifierConfig(CONFIGURED);
    assert.equal(cfg.provider, 'typesafe');
    assert.equal(cfg.baseUrl, CLASSIFIERS.typesafe.base);
    assert.equal(cfg.model, CLASSIFIERS.typesafe.defaultModel);
    assert.equal(cfg.keyEnv, 'TYPESAFE_TEST_KEY');
  });

  test('an unknown provider is not a destination', () => {
    assert.equal(classifierConfig({ classifier: { provider: 'nope' } }), null);
  });

  // The whole privacy pitch: a manifest holds a NAME, never a value.
  test('the block carries no key', () => {
    const cfg = readManifest(join(TEMPLATES, 'agent.yaml'));
    assert.equal(cfg.classifier, null);
    const parsed = classifierConfig(CONFIGURED);
    assert.equal(parsed.key, undefined);
    assert.ok(!Object.values(parsed).includes('sk-live'));
  });

  test('its key env is one loadEnv fills', () => {
    assert.ok(keyEnvs(CONFIGURED).includes('TYPESAFE_TEST_KEY'));
    assert.ok(!keyEnvs(BASE).includes('TYPESAFE_TEST_KEY'));
  });
});

describe('the request', () => {
  test('options are the installed agents, never a hard-coded list', async () => {
    const fetchImpl = endpoint(answered('ui-editor'));
    await withKey('sk-test', () =>
      fastClassify({ task: 'restyle the header', agents: AGENTS, manifest: CONFIGURED, fetchImpl }));

    const { body, url, init } = fetchImpl.calls[0];
    assert.equal(url, 'https://api.typesafe.ai/v1/systemone');
    assert.equal(init.headers.authorization, 'Bearer sk-test');
    assert.deepEqual(body.questions.tier.options, AGENTS.map((a) => a.name));
    assert.equal(body.questions.tier.type, 'choice');
  });

  test('state is structured, and carries the repo state a tier is chosen from', () => {
    const state = stateFor({
      task: 'add a button', buildGreen: false, files: ['a.js', 'b.css'], agents: AGENTS, duties: DUTIES,
    });
    assert.equal(state.task, 'add a button');
    assert.equal(state.build, 'red');
    assert.equal(state.file_count, 2);
    assert.ok(state.entry_rules.length);
    // An agent describes itself here too — the same fields the ladder routes on.
    const fixer = state.agents.find((a) => a.repairs_build);
    assert.ok(fixer, 'the build repairer is declared, not named in code');
    assert.ok(state.agents.some((a) => a.terminal));
  });

  test('an unknown build state is not a red build', () => {
    assert.equal(stateFor({ task: 't', buildGreen: null }).build, 'unknown');
    assert.equal(stateFor({ task: 't', buildGreen: true }).build, 'green');
  });
});

describe('the answer', () => {
  test('a choice with a distribution becomes a tier and a readable reason', async () => {
    const fetchImpl = endpoint(answered('junior-dev', { 'junior-dev': 0.71, 'ui-editor': 0.22 }, 0.71));
    const out = await withKey('sk-test', () =>
      fastClassify({ task: 'add a button', agents: AGENTS, manifest: CONFIGURED, fetchImpl }));

    assert.equal(out.tier, 'junior-dev');
    assert.equal(out.confidence, 0.71);
    assert.equal(out.source, 'system-one');
    // The runner-up is the useful part of a reason a model did not write.
    assert.match(out.reason, /junior-dev 0\.71/);
    assert.match(out.reason, /ui-editor 0\.22/);
  });

  test('envelope and field names are read defensively', () => {
    assert.equal(answerFor({ answers: { t: { choice: 'a' } } }, 't').value, 'a');
    assert.equal(answerFor({ results: { t: { value: 'b' } } }, 't').value, 'b');
    assert.equal(answerFor({ t: { answer: 'c' } }, 't').value, 'c');
    assert.equal(answerFor({ t: 'd' }, 't').value, 'd');
    assert.equal(answerFor({ t: { nothing: 1 } }, 't'), null);
    assert.equal(answerFor({}, 't'), null);
  });

  // Silence about confidence is not the same as low confidence — reading it as
  // zero would bump every single task up a tier.
  test('a missing confidence is full confidence, not none', async () => {
    const fetchImpl = endpoint({ answers: { tier: { choice: 'junior-dev' } } });
    const out = await withKey('sk-test', () =>
      fastClassify({ task: 't', agents: AGENTS, manifest: CONFIGURED, fetchImpl }));
    assert.equal(out.confidence, 1);
    assert.equal(out.tier, 'junior-dev');
  });

  test('an option outside the set we supplied is refused', async () => {
    const fetchImpl = endpoint(answered('principal-engineer'));
    const notices = [];
    const out = await withKey('sk-test', () => fastClassify({
      task: 't', agents: AGENTS, manifest: CONFIGURED, fetchImpl, onNotice: (m) => notices.push(m),
    }));
    assert.equal(out, null);
    assert.equal(notices.length, 1);
  });
});

describe('it is allowed to fail', () => {
  const cases = {
    'no classifier configured': { manifest: BASE, key: 'sk-test', fetchImpl: endpoint(answered('junior-dev')) },
    'no key set':               { manifest: CONFIGURED, key: null, fetchImpl: endpoint(answered('junior-dev')) },
    'a 401':                    { manifest: CONFIGURED, key: 'sk-bad', fetchImpl: endpoint({ error: 'nope' }, { status: 401 }) },
    'a 500':                    { manifest: CONFIGURED, key: 'sk-test', fetchImpl: endpoint({}, { status: 500 }) },
    'a shape nobody knows':     { manifest: CONFIGURED, key: 'sk-test', fetchImpl: endpoint({ weird: true }) },
  };

  for (const [name, { manifest, key, fetchImpl }] of Object.entries(cases)) {
    test(`${name} returns null rather than throwing`, async () => {
      const out = await withKey(key, () =>
        fastClassify({ task: 't', agents: AGENTS, manifest, fetchImpl }));
      assert.equal(out, null);
    });
  }

  test('a network error is caught, and never carries the key', async () => {
    const fetchImpl = async () => { throw new Error('connect ECONNREFUSED with sk-secret-value'); };
    const notices = [];
    const out = await withKey('sk-secret-value', () => fastClassify({
      task: 't', agents: AGENTS, manifest: CONFIGURED, fetchImpl, onNotice: (m) => notices.push(m),
    }));
    assert.equal(out, null);
    assert.ok(notices.length);
    assert.ok(!notices.join(' ').includes('sk-secret-value'), 'the key reached a notice');
    assert.match(notices[0], /\[redacted\]/);
  });

  test('a slow classifier is abandoned, not waited on', async () => {
    const fetchImpl = (url, init) => new Promise((_, reject) => {
      init.signal.addEventListener('abort', () => reject(new Error('aborted')));
    });
    const out = await withKey('sk-test', () =>
      fastClassify({ task: 't', agents: AGENTS, manifest: CONFIGURED, fetchImpl, timeout: 10 }));
    assert.equal(out, null);
  });
});

describe('classify() prefers it, and survives it', () => {
  const run = (manifest, fast, text = JSON.stringify({ tier: 'senior-dev', confidence: 0.9, reason: 'model' })) => {
    const call = stub(text);
    return classify({
      task: 'add a button', manifest, agents: AGENTS, duties: DUTIES, call, fast,
    }).then((r) => ({ ...r, modelCalls: call.calls.length }));
  };

  test('a usable answer means the chat model is never called', async () => {
    const out = await run(CONFIGURED, async () => ({
      tier: 'ui-editor', confidence: 0.88, reason: 'fast', source: 'system-one',
    }));
    assert.equal(out.tier, 'ui-editor');
    assert.equal(out.source, 'system-one');
    assert.equal(out.modelCalls, 0, 'both classifiers ran; that is a double bill');
  });

  test('declining hands the task to the model, unchanged', async () => {
    const out = await run(CONFIGURED, async () => null);
    assert.equal(out.tier, 'senior-dev');
    assert.equal(out.source, 'model');
    assert.equal(out.modelCalls, 1);
  });

  // The two short-circuits are cheaper than any classifier. They stay in front.
  test('a pinned entry tier calls nothing at all', async () => {
    let asked = 0;
    const out = await classify({
      task: 't', manifest: { ...CONFIGURED, entry: 'junior-dev' }, agents: AGENTS, duties: DUTIES,
      call: stub('{}'), fast: async () => { asked++; return null; },
    });
    assert.equal(out.source, 'config');
    assert.equal(asked, 0);
  });

  test('a red build calls nothing at all', async () => {
    let asked = 0;
    const out = await classify({
      task: 't', buildGreen: false, manifest: CONFIGURED, agents: AGENTS, duties: DUTIES,
      call: stub('{}'), fast: async () => { asked++; return null; },
    });
    assert.equal(out.source, 'repo-state');
    assert.equal(asked, 0);
  });

  /**
   * The reason to do any of this. A calibrated probability under the floor
   * must bump exactly like a self-reported one, through the same code.
   */
  test('a calibrated confidence goes through the same floor as the model’s', async () => {
    const out = await run({ ...CONFIGURED, confidenceFloor: 0.6 }, async () => ({
      tier: 'junior-dev', confidence: 0.4, reason: 'close call', source: 'system-one',
    }));
    assert.equal(out.source, 'floor-bump');
    assert.equal(out.classified, 'junior-dev');
    assert.equal(out.tier, 'senior-dev', 'bumped to what junior-dev itself declares');
    assert.match(out.reason, /below floor 0\.6/);
  });

  test('a terminal agent under the floor stays put', async () => {
    const out = await run(CONFIGURED, async () => ({
      tier: 'senior-dev', confidence: 0.1, reason: 'unsure', source: 'system-one',
    }));
    assert.equal(out.tier, 'senior-dev');
    assert.equal(out.source, 'system-one');
  });
});

describe('swarm selection', () => {
  // Named nothing like the default pack, for the same reason
  // custom-agents.test.js is: a default name creeping into the code fails here.
  const parallel = [
    { name: 'scout', parallel: true, owns: ['docs/**'], priority: 1 },
    { name: 'archivist', parallel: true, owns: ['data/**'], priority: 2 },
  ];

  test('a subset is one boolean per agent in one request', async () => {
    const fetchImpl = endpoint((body) => ({
      answers: Object.fromEntries(Object.keys(body.questions).map((k, i) => [k, { noul: i === 0 ? 0.9 : 0.1 }])),
    }));
    const picked = await withKey('sk-test', () =>
      fastSwarm({ task: 'restyle and refactor', agents: parallel, manifest: CONFIGURED, fetchImpl }));

    assert.equal(fetchImpl.calls.length, 1, 'one request, not one per agent');
    const asked = Object.values(fetchImpl.calls[0].body.questions);
    assert.equal(asked.length, parallel.length);
    assert.ok(asked.every((q) => q.type === 'noul'));
    assert.deepEqual(picked, [parallel[0].name]);
  });

  // Keys are positional, so a name can never be misspelled back at us.
  test('question keys are positions, not agent names', async () => {
    const fetchImpl = endpoint({ answers: {} });
    await withKey('sk-test', () =>
      fastSwarm({ task: 't', agents: parallel, manifest: CONFIGURED, fetchImpl }));
    assert.deepEqual(Object.keys(fetchImpl.calls[0].body.questions), parallel.map((_, i) => `a${i}`));
  });

  test('no answers understood falls back rather than swarming nobody', async () => {
    const fetchImpl = endpoint({ answers: {} });
    const picked = await withKey('sk-test', () =>
      fastSwarm({ task: 't', agents: parallel, manifest: CONFIGURED, fetchImpl }));
    assert.equal(picked, null);
  });

  test('selectSwarm asks it first and falls back to the model', async () => {
    const call = stub(JSON.stringify({ agents: [parallel[0].name] }));
    const out = await selectSwarm({
      task: 't', agents: parallel, manifest: CONFIGURED, call, fast: async () => null,
    });
    assert.deepEqual(out, [parallel[0].name]);
    assert.equal(call.calls.length, 1);

    const call2 = stub('{}');
    const out2 = await selectSwarm({
      task: 't', agents: parallel, manifest: CONFIGURED, call: call2,
      fast: async () => parallel.map((a) => a.name),
    });
    assert.deepEqual(out2, parallel.map((a) => a.name));
    assert.equal(call2.calls.length, 0);
  });
});

describe('it never decides enforcement', () => {
  /**
   * The line this whole design rests on. A probabilistic model may choose who
   * works on a task; it may not decide whether a guard fires, whether a human
   * is asked, or whether two scopes are safe to run at once. Those are
   * deterministic and must stay that way, so nothing in the enforcement path
   * is allowed to import this module.
   */
  test('no enforcement module imports the classifier', () => {
    for (const file of ['hooks.js', 'tools.js', 'verify.js', 'session.js']) {
      const src = readFileSync(join(import.meta.dirname, '..', 'src', file), 'utf8');
      assert.ok(!src.includes('classify-fast'), `${file} imports the System One classifier`);
    }
  });
});

describe('adding a Jev key, the way other providers are added', () => {
  const okRes = (body, status = 200) => ({
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
    text: async () => JSON.stringify(body),
  });

  test('people type the model name, not the company', () => {
    assert.equal(classifierByName('jev'), 'typesafe');
    assert.equal(classifierByName('JEV'), 'typesafe');
    assert.equal(classifierByName('typesafe'), 'typesafe');
    assert.equal(classifierByName('system-one'), 'typesafe');
    assert.equal(classifierByName('groq'), null, 'a chat provider is not a router');
    assert.equal(classifierByName(''), null);
  });

  test('a working key reports the model it reaches', async () => {
    const seen = [];
    const fetchImpl = async (url, init) => { seen.push({ url, init }); return okRes({ answers: { ok: { choice: 'yes' } } }); };
    const found = await verifyKey({ key: 'sk-good', fetchImpl });
    assert.equal(found.model, 'jev-latest');
    assert.equal(found.keyEnv, 'TYPESAFE_API_KEY');
    assert.equal(seen[0].url, 'https://api.typesafe.ai/v1/systemone');
    assert.equal(seen[0].init.headers.authorization, 'Bearer sk-good');
  });

  // The key being checked has not been saved yet, so it must not be read from
  // the environment the way a configured one is.
  test('the key under test comes from the argument, not process.env', async () => {
    const had = process.env.TYPESAFE_API_KEY;
    process.env.TYPESAFE_API_KEY = 'sk-a-different-saved-key';
    try {
      let sent = null;
      await verifyKey({ key: 'sk-the-one-being-checked', fetchImpl: async (u, i) => { sent = i.headers.authorization; return okRes({}); } });
      assert.equal(sent, 'Bearer sk-the-one-being-checked');
    } finally {
      if (had === undefined) delete process.env.TYPESAFE_API_KEY; else process.env.TYPESAFE_API_KEY = had;
    }
  });

  test('a refused key is a KeyRejected, not a generic failure', async () => {
    await assert.rejects(
      () => verifyKey({ key: 'sk-bad', fetchImpl: async () => okRes({ error: 'nope' }, 401) }),
      (e) => e instanceof KeyRejected,
    );
  });

  test('a server error is not a refused key', async () => {
    await assert.rejects(
      () => verifyKey({ key: 'sk-x', fetchImpl: async () => okRes({}, 500) }),
      (e) => !(e instanceof KeyRejected) && /500/.test(e.message),
    );
  });

  test('a network failure never carries the key', async () => {
    await assert.rejects(
      () => verifyKey({ key: 'sk-secret-abc', fetchImpl: async () => { throw new Error('getaddrinfo sk-secret-abc'); } }),
      (e) => !e.message.includes('sk-secret-abc') && /\[redacted\]/.test(e.message),
    );
  });
});

describe('setClassifier writes the block without losing the file', () => {
  const scratch = () => {
    const d = mkdtempSync(join(tmpdir(), 'jr-cls-'));
    const f = join(d, 'agent.yaml');
    cpSync(join(TEMPLATES, 'agent.yaml'), f);
    return f;
  };

  test('it appends an active block and readManifest sees it', () => {
    const f = scratch();
    setClassifier({ provider: 'typesafe', keyEnv: 'TYPESAFE_API_KEY' }, f);
    const m = readManifest(f);
    assert.equal(m.classifier.provider, 'typesafe');
    assert.equal(m.classifier.keyEnv, 'TYPESAFE_API_KEY');
    assert.equal(classifierConfig(m).model, 'jev-latest');
  });

  // The comments in agent.yaml are half its documentation, so a write that
  // round-tripped through the parser would be a regression.
  test('every comment in the file survives', () => {
    const f = scratch();
    const before = readFileSync(f, 'utf8').split('\n').filter((l) => l.trim().startsWith('#'));
    setClassifier({ provider: 'typesafe', keyEnv: 'TYPESAFE_API_KEY' }, f);
    const after = readFileSync(f, 'utf8').split('\n').filter((l) => l.trim().startsWith('#'));
    assert.deepEqual(after, before, 'a comment was dropped');
  });

  test('writing twice replaces, never stacks', () => {
    const f = scratch();
    setClassifier({ provider: 'typesafe', keyEnv: 'TYPESAFE_API_KEY' }, f);
    setClassifier({ provider: 'typesafe', keyEnv: 'OTHER_KEY' }, f);
    const text = readFileSync(f, 'utf8');
    assert.equal(text.split('\n').filter((l) => /^\s{2}classifier:\s*$/.test(l)).length, 1);
    assert.equal(readManifest(f).classifier.keyEnv, 'OTHER_KEY');
  });

  test('null removes it and routing goes back to the model', () => {
    const f = scratch();
    setClassifier({ provider: 'typesafe', keyEnv: 'TYPESAFE_API_KEY' }, f);
    setClassifier(null, f);
    assert.equal(readManifest(f).classifier, null);
    assert.equal(classifierConfig(readManifest(f)), null);
  });

  // The whole routing block must still parse, or every run breaks.
  test('the rest of routing is untouched', () => {
    const f = scratch();
    const before = readManifest(f);
    setClassifier({ provider: 'typesafe', keyEnv: 'TYPESAFE_API_KEY' }, f);
    const after = readManifest(f);
    for (const k of ['entry', 'defaultAttempts', 'confidenceFloor', 'degradedFallback']) {
      assert.deepEqual(after[k], before[k], `routing.${k} changed`);
    }
  });
});

describe('a router key is recognised by its own prefix', () => {
  test('jv_live_ is a TypeSafe router key', () => {
    assert.equal(detectClassifier('jv_live_xjYtTseB0000'), 'typesafe');
    assert.equal(detectClassifier('jv_test_abc'), 'typesafe');
  });

  // The reason this check runs BEFORE detectProvider: getting it wrong sends a
  // router key to a model provider and destroys the working key it replaced.
  test('no chat provider key is mistaken for a router key', () => {
    for (const k of ['sk-ant-api03-x', 'gsk_abc', 'sk-or-v1-x', 'xai-abc', 'AIzaSyAbc', 'sk-proj-abc']) {
      assert.equal(detectClassifier(k), null, `${k} was read as a router key`);
    }
  });

  test('and no router key is mistaken for a chat provider key', () => {
    assert.equal(detectProvider('jv_live_xjYtTseB0000'), null);
  });

  test('an unrecognised shape is not guessed at', () => {
    assert.equal(detectClassifier('mystery-key'), null);
    assert.equal(detectClassifier(''), null);
    assert.equal(detectClassifier(null), null);
  });

  // Naming it still works for any shape, which is what makes the prefix a
  // convenience rather than the only way in.
  test('naming it works whatever the key looks like', () => {
    assert.equal(classifierByName('jev'), 'typesafe');
  });

  test('the .env template offers the router variable too', () => {
    const t = envTemplate();
    assert.match(t, /# TYPESAFE_API_KEY=/);
    assert.match(t, /only picks which agent takes a task/);
  });
});
