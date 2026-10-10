import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, writeFileSync, rmSync, readFileSync, realpathSync, cpSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { run } from '../src/run.js';
import { TEMPLATES } from '../src/paths.js';

/**
 * Swarm as waves: retries, escalation, and the shared ledger.
 *
 * A swarm used to be one shot — every agent got a single attempt and a failure
 * simply dropped, while the ladder beside it retried and escalated. These
 * tests pin the two halves of closing that gap: an agent gets the budget IT
 * declares, and what one agent learned is in the brief the next one reads.
 *
 * The invariant that must survive all of it: no two agents whose scopes are
 * not provably disjoint ever run at the same time.
 */

let seq = 0;

/**
 * A repo whose agents declare their own attempts and successor.
 *
 * `fixer` deliberately has NO `owns` and is not parallel: it claims the whole
 * tree, so it must never be run beside a sibling. That is the case the wave
 * planner has to get right.
 */
function sandbox(agents, { test: testScript = 'node -e "process.exit(0)"' } = {}) {
  const root = realpathSync(mkdtempSync(join(tmpdir(), 'jra-wave-')));
  const git = (...a) => execFileSync('git', a, { cwd: root, stdio: 'pipe' });

  writeFileSync(join(root, 'package.json'), JSON.stringify({ name: 'demo', scripts: { test: testScript } }, null, 2));
  for (const d of ['src/api', 'src/ui']) mkdirSync(join(root, d), { recursive: true });
  writeFileSync(join(root, 'src/api/handler.js'), 'export const api = 1;\n');
  writeFileSync(join(root, 'src/ui/view.js'), 'export const ui = 1;\n');

  const dir = join(root, '.gitagent');
  cpSync(TEMPLATES, dir, { recursive: true });
  rmSync(join(dir, 'agents'), { recursive: true, force: true });
  for (const a of agents) {
    mkdirSync(join(dir, 'agents', a.name), { recursive: true });
    const fm = ['---', `name: ${a.name}`, `role: ${a.role ?? a.name}`, `priority: ${a.priority ?? 10}`];
    if (a.parallel) fm.push('parallel: true');
    if (a.attempts) fm.push(`attempts: ${a.attempts}`);
    if (a.escalatesTo) fm.push(`escalates_to: ${a.escalatesTo}`);
    if (a.terminal) fm.push('terminal: true');
    if (a.owns?.length) {
      fm.push('owns:');
      for (const o of a.owns) fm.push(`  - "${o}"`);
    }
    fm.push('---', '', `# ${a.name}`, '');
    writeFileSync(join(dir, 'agents', a.name, 'SOUL.md'), fm.join('\n'));
  }
  writeFileSync(join(root, '.gitignore'), '.gitagent/.env\n.gitagent/.session/\n');
  git('init', '-q');
  git('-c', 'user.email=t@t', '-c', 'user.name=t', 'add', '--all');
  git('-c', 'user.email=t@t', '-c', 'user.name=t', 'commit', '-q', '-m', 'init');
  return { root, dir };
}

async function inRepo(box, fn) {
  const prev = process.cwd();
  process.chdir(box.root);
  try { return await fn(); } finally {
    process.chdir(prev);
    rmSync(box.root, { recursive: true, force: true });
  }
}

/**
 * A model that answers as whichever agent the system prompt names, and records
 * what each one was briefed with.
 *
 * `waves` records the set of agents live at the same moment, which is how the
 * concurrency invariant is checked: the planner's promise is about who runs
 * together, not about who runs.
 */
function scripted(byAgent) {
  const calls = [];
  const briefs = {};
  const live = new Set();
  const overlaps = [];

  const fn = async (_manifest, req) => {
    if (/handing it to someone else/.test(req.system ?? '')) {
      return { text: '{"completed":[]}', toolCalls: [], stopReason: 'end_turn' };
    }
    const who = Object.keys(byAgent).find((n) => req.system?.includes('You are `' + n + '`'));
    if (!who) return { text: '', toolCalls: [], stopReason: 'end_turn' };

    const used = calls.filter((x) => x === who).length;
    if (used === 0) briefs[who] = req.messages?.[0]?.content ?? '';
    calls.push(who);

    live.add(who);
    if (live.size > 1) overlaps.push([...live].sort().join('+'));
    await new Promise((r) => setTimeout(r, 1));
    live.delete(who);

    const t = (byAgent[who] ?? [])[used] ?? {};
    return {
      text: t.text ?? '',
      toolCalls: (t.tools ?? []).map(([name, input]) => ({ id: `c${++seq}`, name, input })),
      stopReason: t.tools ? 'tool_use' : 'end_turn',
    };
  };
  fn.calls = calls;
  fn.briefs = briefs;
  fn.overlaps = overlaps;
  fn.count = (name) => calls.filter((x) => x === name).length;
  return fn;
}

const tool = (name, input) => [name, input];
const read = (root, p) => readFileSync(join(root, p), 'utf8');
/** Enough non-tool replies to burn an attempt, whatever the budget. */
const stuck = Array.from({ length: 6 }, () => ({ text: 'stuck' }));

const PAIR = [
  { name: 'api-dev', parallel: true, owns: ['src/api/**'], attempts: 2, escalatesTo: 'fixer' },
  { name: 'ui-dev', parallel: true, owns: ['src/ui/**'], attempts: 1 },
  { name: 'fixer', attempts: 1, terminal: true, role: 'takes over anything' },
];

describe('an agent gets the attempts it declares', () => {
  test('a first failure is retried inside the swarm, not dropped', async () => {
    const box = sandbox(PAIR);
    await inRepo(box, async () => {
      const call = scripted({
        'api-dev': [
          { text: 'no tool this time' },
          { tools: [tool('write_file', { path: 'src/api/handler.js', content: 'export const api = 2;\n' })] },
          { tools: [tool('done', { summary: 'api updated on the retry' })] },
        ],
        'ui-dev': [
          { tools: [tool('write_file', { path: 'src/ui/view.js', content: 'export const ui = 2;\n' })] },
          { tools: [tool('done', { summary: 'ui updated' })] },
        ],
      });
      const out = await run(['update both layers'], { swarm: true }, { call });

      assert.equal(out.status, 'done', out.reason ?? '');
      assert.match(read(box.root, 'src/api/handler.js'), /api = 2/, 'the retry never ran');
      assert.match(read(box.root, 'src/ui/view.js'), /ui = 2/);
      assert.equal(call.count('fixer'), 0, 'it escalated instead of using its second attempt');
    });
  });

  test('an agent that declares one attempt does not get two', async () => {
    const box = sandbox([
      { name: 'api-dev', parallel: true, owns: ['src/api/**'], attempts: 1 },
      { name: 'ui-dev', parallel: true, owns: ['src/ui/**'], attempts: 1 },
    ]);
    await inRepo(box, async () => {
      const call = scripted({
        'api-dev': stuck,
        'ui-dev': [
          { tools: [tool('write_file', { path: 'src/ui/view.js', content: 'export const ui = 2;\n' })] },
          { tools: [tool('done', { summary: 'ui updated' })] },
        ],
      });
      const out = await run(['update both layers'], { swarm: true }, { call });

      assert.equal(call.count('api-dev'), 1, 'a one-attempt agent ran more than once');
      assert.equal(out.status, 'partial');
      // The sibling's work survives its neighbour's failure, as before.
      assert.match(read(box.root, 'src/ui/view.js'), /ui = 2/);
    });
  });
});

describe('a spent agent escalates to the successor it names', () => {
  test('its heir runs, and the swarm finishes', async () => {
    const box = sandbox(PAIR);
    await inRepo(box, async () => {
      const call = scripted({
        'api-dev': stuck,
        'ui-dev': [
          { tools: [tool('write_file', { path: 'src/ui/view.js', content: 'export const ui = 2;\n' })] },
          { tools: [tool('done', { summary: 'ui updated' })] },
        ],
        fixer: [
          { tools: [tool('write_file', { path: 'src/api/handler.js', content: 'export const api = 3;\n' })] },
          { tools: [tool('done', { summary: 'fixer finished the api work' })] },
        ],
      });
      const out = await run(['update both layers'], { swarm: true }, { call });

      assert.equal(call.count('api-dev'), 2, 'it should spend both attempts before escalating');
      assert.equal(call.count('fixer'), 2, 'the heir never ran');
      assert.match(read(box.root, 'src/api/handler.js'), /api = 3/);
      assert.equal(out.status, 'done', out.reason ?? '');
    });
  });

  /**
   * The point of the whole exercise. `fixer` is briefed from the ledger, which
   * by then holds what api-dev recorded AND what ui-dev recorded — so it does
   * not start from the bare task.
   */
  test('the heir is briefed from the shared ledger, not the raw task', async () => {
    const box = sandbox(PAIR);
    await inRepo(box, async () => {
      const call = scripted({
        'api-dev': stuck,
        'ui-dev': [
          { tools: [tool('write_file', { path: 'src/ui/view.js', content: 'export const ui = 2;\n' })] },
          { tools: [tool('done', { summary: 'ui updated' })] },
        ],
        fixer: [{ tools: [tool('done', { summary: 'read the record' })] }],
      });
      await run(['update both layers'], { swarm: true }, { call });

      const brief = call.briefs.fixer ?? '';
      assert.notEqual(brief, 'update both layers', 'the heir got the bare task');
      // Task and objective are never trimmed out of a compiled brief.
      assert.match(brief, /update both layers/);
      assert.match(brief, /## /, 'a compiled brief has sections; this is not one');
    });
  });

  test('a cycle between two agents terminates', async () => {
    const box = sandbox([
      { name: 'a-dev', parallel: true, owns: ['src/api/**'], attempts: 1, escalatesTo: 'b-dev' },
      { name: 'b-dev', parallel: true, owns: ['src/ui/**'], attempts: 1, escalatesTo: 'a-dev' },
    ]);
    await inRepo(box, async () => {
      const call = scripted({ 'a-dev': stuck, 'b-dev': stuck });
      const out = await run(['do something neither can do'], { swarm: true }, { call });

      assert.equal(out.status, 'stopped');
      // One attempt each and no more: neither may escalate into an agent that
      // has already spent its own budget.
      assert.equal(call.count('a-dev'), 1);
      assert.equal(call.count('b-dev'), 1);
    });
  });
});

describe('scope still decides who may run together', () => {
  /**
   * The safety rule a wave loop could easily break. The original group was
   * checked by swarmable(), but an escalation target was never in that check —
   * and `fixer` declares no scope, so it claims the whole tree. It must run on
   * its own.
   */
  test('an heir with no declared scope never runs beside a sibling', async () => {
    const box = sandbox([
      { name: 'api-dev', parallel: true, owns: ['src/api/**'], attempts: 1, escalatesTo: 'fixer' },
      { name: 'ui-dev', parallel: true, owns: ['src/ui/**'], attempts: 1, escalatesTo: 'fixer' },
      { name: 'fixer', attempts: 2, terminal: true },
    ]);
    await inRepo(box, async () => {
      const call = scripted({
        'api-dev': stuck,
        'ui-dev': stuck,
        fixer: [
          { tools: [tool('write_file', { path: 'src/api/handler.js', content: 'export const api = 9;\n' })] },
          { tools: [tool('done', { summary: 'fixer took over' })] },
        ],
      });
      await run(['both fail and hand over'], { swarm: true }, { call });

      assert.ok(call.count('fixer') > 0, 'the heir never ran');
      // The two scoped agents may overlap with each other; the unscoped heir
      // may overlap with nobody.
      for (const pair of call.overlaps) {
        assert.ok(!pair.includes('fixer'), `fixer ran concurrently: ${pair}`);
      }
    });
  });

  test('two siblings escalating to one heir queue it once', async () => {
    const box = sandbox([
      { name: 'api-dev', parallel: true, owns: ['src/api/**'], attempts: 1, escalatesTo: 'fixer' },
      { name: 'ui-dev', parallel: true, owns: ['src/ui/**'], attempts: 1, escalatesTo: 'fixer' },
      { name: 'fixer', attempts: 1, terminal: true },
    ]);
    await inRepo(box, async () => {
      const call = scripted({
        'api-dev': stuck,
        'ui-dev': stuck,
        fixer: [{ tools: [tool('done', { summary: 'one turn only' })] }],
      });
      await run(['both hand over to the same agent'], { swarm: true }, { call });

      // One attempt declared means one turn, even though two agents asked for
      // it. Queueing it twice would run the same agent against itself.
      assert.equal(call.count('fixer'), 1);
    });
  });

  test('the scoped pair still do run at the same time', async () => {
    const box = sandbox([
      { name: 'api-dev', parallel: true, owns: ['src/api/**'], attempts: 1 },
      { name: 'ui-dev', parallel: true, owns: ['src/ui/**'], attempts: 1 },
    ]);
    await inRepo(box, async () => {
      const call = scripted({
        'api-dev': [
          { tools: [tool('write_file', { path: 'src/api/handler.js', content: 'export const api = 2;\n' })] },
          { tools: [tool('done', { summary: 'api' })] },
        ],
        'ui-dev': [
          { tools: [tool('write_file', { path: 'src/ui/view.js', content: 'export const ui = 2;\n' })] },
          { tools: [tool('done', { summary: 'ui' })] },
        ],
      });
      await run(['update both layers'], { swarm: true }, { call });
      assert.ok(call.overlaps.includes('api-dev+ui-dev'), 'the swarm serialised a safe pair');
    });
  });
});
