/**
 * scripts/autoclose-labeled-issues.js
 *
 * Closes open issues that carry the "autoclose" label and have been inactive
 * (no updates) for more than INACTIVITY_DAYS days.
 *
 * Required environment variables:
 *   GITHUB_TOKEN    – GitHub Actions token (or PAT with repo:issues write access)
 *   REPO_OWNER      – Repository owner (e.g. VectifyAI)
 *   REPO_NAME       – Repository name  (e.g. PageIndex)
 *
 * Optional environment variables:
 *   INACTIVITY_DAYS – Days of inactivity before closing (default: 7)
 *   DRY_RUN         – If "true", report but do not close issues (default: false)
 */

'use strict';

const https = require('https');

// ── Configuration ─────────────────────────────────────────────────────────────

const GITHUB_TOKEN    = process.env.GITHUB_TOKEN;
const REPO_OWNER      = process.env.REPO_OWNER;
const REPO_NAME       = process.env.REPO_NAME;
const INACTIVITY_DAYS = parseInt(process.env.INACTIVITY_DAYS || '7', 10);
const DRY_RUN         = process.env.DRY_RUN === 'true';

// ── HTTP helper ───────────────────────────────────────────────────────────────

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
        'User-Agent':    'PageIndex-Autoclose-Script/1.0',
        'X-GitHub-Api-Version': '2022-11-28',
        ...(payload ? {
          'Content-Type':   'application/json',
          'Content-Length': Buffer.byteLength(payload),
        } : {}),
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

/** Simple sleep helper for rate-limiting. */
const sleep = (ms) => new Promise(r => setTimeout(r, ms));

// ── Core logic ────────────────────────────────────────────────────────────────

/**
 * Fetches all open issues with the "autoclose" label, paginating as needed.
 */
async function fetchAutocloseIssues() {
  const issues = [];
  let page = 1;
  while (true) {
    const data = await githubRequest(
      'GET',
      `/repos/${REPO_OWNER}/${REPO_NAME}/issues?state=open&labels=autoclose&per_page=100&page=${page}`
    );
    if (!Array.isArray(data) || data.length === 0) break;
    // Filter out any pull requests that may surface
    issues.push(...data.filter(i => !i.pull_request));
    if (data.length < 100) break;
    page++;
  }
  return issues;
}

/**
 * Closes a single issue with a polite explanatory comment.
 */
async function closeIssue(issueNumber, inactivityDays) {
  const body =
    `This issue has been automatically closed because it was marked as a **duplicate** ` +
    `and has had no new activity for ${inactivityDays} day(s).\n\n` +
    `If you believe this was closed in error, please reopen the issue and leave a comment. ` +
    `New human activity will prevent automatic closure in the future.\n\n` +
    `Thank you for your contribution! 🙏`;

  // Post closing comment first
  await githubRequest(
    'POST',
    `/repos/${REPO_OWNER}/${REPO_NAME}/issues/${issueNumber}/comments`,
    { body }
  );

  // Close the issue
  await githubRequest(
    'PATCH',
    `/repos/${REPO_OWNER}/${REPO_NAME}/issues/${issueNumber}`,
    { state: 'closed', state_reason: 'not_planned' }
  );
}

// ── Entry point ───────────────────────────────────────────────────────────────

async function main() {
  // Validate required env vars
  const missing = ['GITHUB_TOKEN', 'REPO_OWNER', 'REPO_NAME']
    .filter(k => !process.env[k]);
  if (missing.length) {
    console.error(`Missing required environment variables: ${missing.join(', ')}`);
    process.exit(1);
  }

  const cutoff = new Date(Date.now() - INACTIVITY_DAYS * 24 * 60 * 60 * 1000);

  console.log(`Auto-close inactive labelled issues`);
  console.log(`  Repository:      ${REPO_OWNER}/${REPO_NAME}`);
  console.log(`  Inactivity days: ${INACTIVITY_DAYS}  (cutoff: ${cutoff.toISOString()})`);
  console.log(`  Dry run:         ${DRY_RUN}`);

  const issues = await fetchAutocloseIssues();
  console.log(`\nFound ${issues.length} open issue(s) with "autoclose" label.`);

  let closedCount = 0;
  let skippedCount = 0;

  for (const issue of issues) {
    const lastActivity = new Date(issue.updated_at);
    const inactive     = lastActivity < cutoff;
    const daysSince    = Math.floor((Date.now() - lastActivity.getTime()) / (1000 * 60 * 60 * 24));

    if (!inactive) {
      console.log(`  #${issue.number} — active ${daysSince}d ago, skipping.`);
      skippedCount++;
      continue;
    }

    console.log(`  #${issue.number} — inactive for ${daysSince}d: "${issue.title}"`);

    if (DRY_RUN) {
      console.log(`    [DRY RUN] Would close issue #${issue.number}`);
      closedCount++;
      continue;
    }

    try {
      await closeIssue(issue.number, INACTIVITY_DAYS);
      console.log(`    ✅ Closed issue #${issue.number}`);
      closedCount++;
    } catch (err) {
      console.error(`    ❌ Failed to close #${issue.number}: ${err.message}`);
    }

    // Respect GitHub's secondary rate limit
    await sleep(1000);
  }

  console.log(`\nSummary: ${closedCount} closed, ${skippedCount} still active.`);
}

main().catch(err => {
  console.error('Fatal error:', err.message);
  process.exit(1);
});
