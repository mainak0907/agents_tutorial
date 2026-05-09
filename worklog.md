# JIRA Work Log Automation

A Python script that automates repetitive JIRA work logging tasks by reading input from an Excel file. For each row, it creates three subtasks under a parent JIRA ticket, assigns the ticket, marks it as Done, sets a fix version, and logs work on each subtask individually.

---

## What It Does

For every row in the Excel input file, the script performs the following actions in order:

1. Fetches the parent JIRA ticket to retrieve its summary and project key
2. Resolves the assignee's display name to a JIRA account ID
3. Resolves the fix version name to its internal JIRA ID
4. Assigns the parent ticket to the specified user
5. Sets the fix version on the parent ticket
6. Transitions the parent ticket to **Done**
7. Creates **3 subtasks** under the parent:
   - `[ANALYSIS] <parent summary>`
   - `[DEVELOPMENT] <parent summary>`
   - `[TESTING] <parent summary>`
8. Logs work on each subtask with its own hours, date, and comment

Errors on a single row are caught and reported at the end without stopping the rest of the rows from being processed.

---

## Project Structure

```
├── jira_automation.py          # Main automation script
├── jira_worklog_template.xlsx  # Excel input template (fill this)
├── .env.template               # Copy to .env and fill in credentials
└── README.md
```

---

## Prerequisites

- Python 3.8 or higher
- A JIRA Cloud account with API access
- A JIRA API token — generate one at:
  `https://id.atlassian.com/manage-profile/security/api-tokens`

---

## Installation

**1. Clone or download the project files**

**2. Install dependencies**
```bash
pip install requests pandas openpyxl python-dotenv
```

**3. Set up your credentials**

Copy `.env.template` to `.env` and fill in your details:
```bash
cp .env.template .env
```

Edit `.env`:
```
JIRA_BASE_URL=https://your-company.atlassian.net
JIRA_EMAIL=you@yourcompany.com
JIRA_API_TOKEN=your_api_token_here
```

> ⚠️ Never commit `.env` to version control. Add it to `.gitignore`.

---

## Excel Input File

Use `jira_worklog_template.xlsx` as your input. Fill one row per parent JIRA ticket.

### Column Reference

| Column | Description | Format / Example |
|---|---|---|
| `jira_number` | Parent JIRA ticket key | `PROJ-101` |
| `assignee` | Full display name of the JIRA user | `John Smith` |
| `fix_version` | Fix version / sprint name exactly as it appears in JIRA | `Sprint 12` |
| `analysis_hours` | Hours to log on the `[ANALYSIS]` subtask | `2` or `1.5` |
| `analysis_date` | Work log date for `[ANALYSIS]` | `2025-05-01` |
| `analysis_comment` | Work log comment for `[ANALYSIS]` | `Reviewed requirements` |
| `development_hours` | Hours to log on the `[DEVELOPMENT]` subtask | `5` |
| `development_date` | Work log date for `[DEVELOPMENT]` | `2025-05-02` |
| `development_comment` | Work log comment for `[DEVELOPMENT]` | `Implemented the feature` |
| `testing_hours` | Hours to log on the `[TESTING]` subtask | `1.5` |
| `testing_date` | Work log date for `[TESTING]` | `2025-05-03` |
| `testing_comment` | Work log comment for `[TESTING]` | `Ran regression tests` |

### Rules
- **Dates** must be in `YYYY-MM-DD` format
- **Hours** can be whole numbers or decimals (e.g. `1.5` = 1 hour 30 minutes)
- **Assignee** must match the display name in JIRA exactly (case-insensitive)
- **Fix version** must match the version name in JIRA exactly (case-insensitive)

---

## Usage

```bash
python jira_automation.py <path_to_excel_file>
```

**Example:**
```bash
python jira_automation.py jira_worklog_template.xlsx
```

### Sample Output

```
============================================================
Processing: PROJ-101
  Summary   : Implement login with OAuth
  Project   : PROJ
  Assignee  : John Smith (abc123xyz)
  Fix Ver.  : Sprint 12 (10020)
  ✔ Assigned and fix version set on PROJ-101
  ✔ PROJ-101 marked as Done
  ✔ Created subtask PROJ-102: [ANALYSIS] Implement login with OAuth
      Logged 2.0h on 2025-05-01 — "Reviewed requirements"
  ✔ Created subtask PROJ-103: [DEVELOPMENT] Implement login with OAuth
      Logged 5.0h on 2025-05-02 — "Implemented the feature"
  ✔ Created subtask PROJ-104: [TESTING] Implement login with OAuth
      Logged 1.5h on 2025-05-03 — "Ran regression tests"
  ✅ Done with PROJ-101
============================================================
All rows processed successfully ✅
```

---

## API Endpoints Used

The script communicates exclusively with the JIRA REST API v2. All requests use Basic Auth (`email:api_token`).

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `GET` | `/rest/api/2/issue/{key}` | Fetch parent issue summary and project |
| `GET` | `/rest/api/2/user/search?query=` | Resolve display name → accountId |
| `GET` | `/rest/api/2/project/{key}/versions` | Resolve fix version name → id |
| `GET` | `/rest/api/2/issue/{key}/transitions` | Get available workflow transitions |
| `PUT` | `/rest/api/2/issue/{key}/assignee` | Assign the parent ticket |
| `PUT` | `/rest/api/2/issue/{key}` | Set fix version on parent ticket |
| `POST` | `/rest/api/2/issue/{key}/transitions` | Transition parent ticket to Done |
| `POST` | `/rest/api/2/issue` | Create each subtask (called 3× per row) |
| `POST` | `/rest/api/2/issue/{key}/worklog` | Log work on each subtask (called 3× per row) |

---

## Error Handling

- If a row fails (e.g. invalid JIRA key, unknown user, missing fix version), it is skipped and the error is printed — other rows continue processing.
- A summary of all errors is printed at the end of the run.
- Common errors and causes:

| Error | Likely Cause |
|-------|-------------|
| `No JIRA user found for '...'` | Display name doesn't match any JIRA user |
| `Fix version '...' not found` | Version name doesn't exist in the project; check spelling |
| `Transition 'Done' not found` | The ticket's workflow uses a different name (e.g. `Closed`, `Resolved`) |
| `404` on issue fetch | The JIRA key doesn't exist or you lack permission |
| `401 Unauthorized` | Invalid email or API token in `.env` |

---

## Security Notes

- Store credentials only in `.env` — never hardcode them in the script
- Add `.env` to `.gitignore` before pushing to any repository
- API tokens can be revoked at any time from your Atlassian account settings
- The script only requires standard JIRA project-level permissions (create issues, log work, transition issues)

---

## Limitations

- Designed for **JIRA Cloud** using API v2; JIRA Server/Data Center may require adjustments
- Assumes the workflow has a transition named exactly `Done` — update `get_transition_id("Done")` in the script if your workflow uses a different name
- Sub-task issue type must be named `Sub-task` in your JIRA project configuration
