/**
 * scripts/backfill-dedupe.js
 *
 * Backfills duplicate detection for historical issues.
 * Fetches issues created within the last DAYS_BACK days, searches for
 * candidate duplicates via the GitHub Search API, and asks the Anthropic
 * API to determine whether each issue is a duplicate.
 *
 * Required environment variables:
 *   GITHUB_TOKEN       – GitHub Actions token (or PAT with repo access)
 *   ANTHROPIC_API_KEY  – Anthropic API key (mapped from AUTHROPIC_API_KEY secret)
 *   REPO_OWNER         – Repository owner (e.g. VectifyAI)
 *   REPO_NAME          – Repository name  (e.g. PageIndex)
 *
 * Optional environment variables:
 *   DAYS_BACK   – How many days back to process   (default: 30)
 *   DRY_RUN     – If "true", analyse but do not write to GitHub (default: false)
 */

'use strict';

const https = require('https');

// ── Configuration ─────────────────────────────────────────────────────────────

const GITHUB_TOKEN     = process.env.GITHUB_TOKEN;
const ANTHROPIC_API_KEY = process.env.ANTHROPIC_API_KEY;
const REPO_OWNER       = process.env.REPO_OWNER;
const REPO_NAME        = process.env.REPO_NAME;
const DAYS_BACK        = parseInt(process.env.DAYS_BACK  || '30', 10);
const DRY_RUN          = process.env.DRY_RUN === 'true';

const STOP_WORDS = new Set([
  'a','an','the','is','in','on','at','to','for','of','and','or','but','not',
  'with','this','that','it','be','are','was','has','have','does','do','how',
  'why','when','where','what','which','who','will','can','could','should',
  'would','may','might','must','get','got','use','using','used','error',
  'issue','bug','feature','request','problem','question','please','just',
  'after','before','during','about','from','into','also','then','than',
]);

// ── HTTP helpers ──────────────────────────────────────────────────────────────

/**
 * Makes an authenticated GitHub REST API request.
 * @param {string} method  HTTP method
 * @param {string} path    API path (e.g. '/repos/owner/repo/issues')
 * @param {object|null} body  Request body (will be JSON-encoded)
 * @returns {Promise<object>}
 */
function githubRequest(method, path, body = null) {
  return new Promise((resolve, reject) => {
    const payload = body ? JSON.stringify(body) : null;
    const options = {
      hostname: 'api.github.com',
      path,
      method,
      headers: {
        'Authorization': `Bearer ${GITHUB_TOKEN}`,
        'Accept':        'application/vnd.github+json',
        'User-Agent':    'PageIndex-Backfill-Script/1.0',
        'X-GitHub-Api-Version': '2022-11-28',
        ...(payload ? { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(payload) } : {}),
      },
    };

    const req = https.request(options, (res) => {
      let data = '';
      res.on('data', chunk => (data += chunk));
      res.on('end', () => {
        if (res.statusCode >= 400) {
          reject(new Error(`GitHub API ${method} ${path} → ${res.statusCode}: ${data}`));
          return;
        }
        try {
          resolve(data ? JSON.parse(data) : {});
        } catch {
          resolve({});
        }
      });
    });
    req.on('error', reject);
    if (payload) req.write(payload);
    req.end();
  });
}

/**
 * Calls the Anthropic Messages API and returns Claude's text response.
 * @param {string} prompt  User prompt
 * @returns {Promise<string>}
 */
function callClaude(prompt) {
  return new Promise((resolve, reject) => {
    const body = JSON.stringify({
      model:      'claude-haiku-4-5',
      max_tokens: 1024,
      messages:   [{ role: 'user', content: prompt }],
    });

    const options = {
      hostname: 'api.anthropic.com',
      path:     '/v1/messages',
      method:   'POST',
      headers:  {
        'Content-Type':      'application/json',
        'Content-Length':    Buffer.byteLength(body),
        'x-api-key':         ANTHROPIC_API_KEY,
        'anthropic-version': '2023-06-01',
      },
    };

    const req = https.request(options, (res) => {
      let data = '';
      res.on('data', chunk => (data += chunk));
      res.on('end', () => {
        try {
          const parsed = JSON.parse(data);
          if (parsed.error) {
            reject(new Error(`Anthropic API error: ${parsed.error.message}`));
            return;
          }
          const text = (parsed.content || [])
            .filter(b => b.type === 'text')
            .map(b => b.text)
            .join('');
          resolve(text);
        } catch (err) {
          reject(new Error(`Failed to parse Anthropic response: ${err.message}`));
        }
      });
    });
    req.on('error', reject);
    req.write(body);
    req.end();
  });
}

/** Simple sleep helper for rate-limiting. */
const sleep = (ms) => new Promise(r => setTimeout(r, ms));

// ── Core logic ────────────────────────────────────────────────────────────────

/**
 * Fetches open issues created since `since` (ISO 8601 string), paginating as needed.
 */
async function fetchIssuesSince(since) {
  const issues = [];
  let page = 1;
  while (true) {
    const data = await githubRequest(
      'GET',
      `/repos/${REPO_OWNER}/${REPO_NAME}/issues?state=open&sort=created&direction=desc&since=${since}&per_page=100&page=${page}`
    );
    if (!Array.isArray(data) || data.length === 0) break;
    // Filter out pull requests
    issues.push(...data.filter(i => !i.pull_request));
    if (data.length < 100) break;
    page++;
  }
  return issues;
}

/**
 * Searches for up to 10 candidate duplicate issues for the given issue.
 */
async function findCandidates(issue) {
  const keywords = (issue.title || '')
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, ' ')
    .split(/\s+/)
    .filter(w => w.length > 2 && !STOP_WORDS.has(w))
    .slice(0, 6)
    .join(' ');

  if (!keywords) return [];

  const q = encodeURIComponent(
    `repo:${REPO_OWNER}/${REPO_NAME} is:issue state:open ${keywords}`
  );

  const data = await githubRequest('GET', `/search/issues?q=${q}&per_page=15`);
  return (data.items || [])
    .filter(item => item.number !== issue.number && !item.pull_request)
    .slice(0, 10);
}

/**
 * Builds the duplicate-detection prompt for Claude.
 */
function buildPrompt(issue, candidates) {
  const candidatesText = candidates
    .map(c => `#${c.number}: ${c.title}\nURL: ${c.html_url}\n${(c.body || '').substring(0, 500)}`)
    .join('\n---\n');

  return `You are a GitHub issue triage assistant.

Analyze whether the following open issue is a duplicate of any of the candidate issues listed below.

== NEW ISSUE #${issue.number} ==
Title: ${issue.title}
Body:
${(issue.body || '(no body)').substring(0, 3000)}

== CANDIDATE ISSUES (up to 10) ==
${candidatesText}

RULES:
- Only flag as a duplicate if you are at least 85% confident.
- A minor difference in wording does NOT make an issue non-duplicate if they describe the same underlying problem or feature request.

Respond with ONLY a JSON object (no markdown, no other text):
{
  "is_duplicate": true or false,
  "duplicate_issues": [array of integer issue numbers that this is a duplicate of, empty if none],
  "explanation": "one or two sentences explaining your reasoning"
}`;
}

/**
 * Parses Claude's JSON response robustly.
 * Returns { is_duplicate, duplicate_issues, explanation } or null on failure.
 */
function parseClaudeResponse(text) {
  // Try to extract a JSON object from the response
  const jsonMatch = text.match(/\{[\s\S]*\}/);
  if (!jsonMatch) return null;
  try {
    const parsed = JSON.parse(jsonMatch[0]);
    return {
      is_duplicate:     Boolean(parsed.is_duplicate),
      duplicate_issues: Array.isArray(parsed.duplicate_issues) ? parsed.duplicate_issues.map(Number) : [],
      explanation:      String(parsed.explanation || ''),
    };
  } catch {
    return null;
  }
}

/**
 * Posts a duplicate-found comment on the issue.
 */
async function postDuplicateComment(issueNumber, duplicateIssueNumbers, explanation) {
  const links = duplicateIssueNumbers
    .map(n => `- #${n}`)
    .join('\n');

  const body =
    `👋 Thank you for taking the time to open this issue!\n\n` +
    `After automated analysis, this issue appears to be a duplicate of:\n\n` +
    `${links}\n\n` +
    `${explanation}\n\n` +
    `Please subscribe to the original issue(s) above to follow updates. ` +
    `This issue will be automatically closed after a short inactivity period.\n\n` +
    `<!-- DEDUPE_RESULT: {"is_duplicate":true,"issues":${JSON.stringify(duplicateIssueNumbers)}} -->`;

  await githubRequest(
    'POST',
    `/repos/${REPO_OWNER}/${REPO_NAME}/issues/${issueNumber}/comments`,
    { body }
  );
}

/**
 * Adds labels to an issue, creating them if they do not exist.
 */
async function ensureLabelAndApply(issueNumber, labelNames) {
  const knownLabels = {
    duplicate: { color: 'cfd3d7', description: 'This issue or pull request already exists' },
    autoclose:  { color: 'e4e669', description: 'Will be auto-closed after a period of inactivity' },
  };

  for (const name of labelNames) {
    try {
      await githubRequest('GET', `/repos/${REPO_OWNER}/${REPO_NAME}/labels/${encodeURIComponent(name)}`);
    } catch {
      const meta = knownLabels[name] || { color: 'ededed', description: '' };
      await githubRequest('POST', `/repos/${REPO_OWNER}/${REPO_NAME}/labels`, { name, ...meta });
    }
  }

  await githubRequest(
    'POST',
    `/repos/${REPO_OWNER}/${REPO_NAME}/issues/${issueNumber}/labels`,
    { labels: labelNames }
  );
}

/**
 * Processes a single issue: finds candidates, asks Claude, and acts on the result.
 */
async function processIssue(issue) {
  const num = issue.number;
  console.log(`\nProcessing issue #${num}: ${issue.title}`);

  // Skip already-labelled issues
  const existingLabels = (issue.labels || []).map(l => l.name);
  if (existingLabels.includes('duplicate')) {
    console.log(`  → Already labelled as duplicate, skipping.`);
    return;
  }

  const candidates = await findCandidates(issue);
  if (candidates.length === 0) {
    console.log(`  → No candidates found, skipping.`);
    return;
  }
  console.log(`  → Found ${candidates.length} candidate(s): ${candidates.map(c => `#${c.number}`).join(', ')}`);

  const prompt   = buildPrompt(issue, candidates);
  const rawReply = await callClaude(prompt);
  const result   = parseClaudeResponse(rawReply);

  if (!result) {
    console.warn(`  ⚠️  Could not parse Claude response for #${num}. Raw:\n${rawReply.substring(0, 300)}`);
    return;
  }

  console.log(`  → is_duplicate=${result.is_duplicate}, issues=${JSON.stringify(result.duplicate_issues)}`);
  console.log(`     ${result.explanation}`);

  if (!result.is_duplicate || result.duplicate_issues.length === 0) {
    console.log(`  → Not a duplicate.`);
    return;
  }

  if (DRY_RUN) {
    console.log(`  [DRY RUN] Would post comment and apply labels to #${num}`);
    return;
  }

  await postDuplicateComment(num, result.duplicate_issues, result.explanation);
  await ensureLabelAndApply(num, ['duplicate', 'autoclose']);
  console.log(`  ✅ Commented and labelled #${num}`);
}

// ── Entry point ───────────────────────────────────────────────────────────────

async function main() {
  // Validate required env vars
  const missing = ['GITHUB_TOKEN', 'ANTHROPIC_API_KEY', 'REPO_OWNER', 'REPO_NAME']
    .filter(k => !process.env[k]);
  if (missing.length) {
    console.error(`Missing required environment variables: ${missing.join(', ')}`);
    process.exit(1);
  }

  const since = new Date(Date.now() - DAYS_BACK * 24 * 60 * 60 * 1000).toISOString();

  console.log(`Backfilling duplicate detection`);
  console.log(`  Repository:  ${REPO_OWNER}/${REPO_NAME}`);
  console.log(`  Days back:   ${DAYS_BACK}  (since ${since})`);
  console.log(`  Dry run:     ${DRY_RUN}`);

  const issues = await fetchIssuesSince(since);
  console.log(`\nFetched ${issues.length} open issue(s) to process.`);

  for (const issue of issues) {
    await processIssue(issue);
    // Respect GitHub and Anthropic rate limits
    await sleep(2500);
  }

  console.log('\nBackfill complete.');
}

main().catch(err => {
  console.error('Fatal error:', err.message);
  process.exit(1);
});
