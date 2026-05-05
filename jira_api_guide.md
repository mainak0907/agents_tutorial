# Jira REST API Guide — fixVersion & Story Points

---

## 1. Authentication — Personal Access Token (PAT)

Jira supports PAT-based authentication for both **Jira Cloud** and **Jira Server/Data Center**.

### Jira Server / Data Center
Use the `Authorization: Bearer` header:
```
Authorization: Bearer YOUR_PERSONAL_ACCESS_TOKEN
```

### Jira Cloud
Use **HTTP Basic Auth** with your email + API token (PATs are server-side; Cloud uses API tokens):
```
Authorization: Basic base64(email:api_token)
```
In Postman, set Auth Type → **Basic Auth** → enter email + API token.

---

## 2. API Endpoints Reference

| Purpose | Method | Endpoint |
|---|---|---|
| Update an issue | `PUT` | `/rest/api/3/issue/{issueKey}` |
| Get project versions | `GET` | `/rest/api/3/project/{projectKey}/versions` |
| Get all fields (find Story Points ID) | `GET` | `/rest/api/3/field` |
| Get editable fields for an issue | `GET` | `/rest/api/3/issue/{issueKey}/editmeta` |

---

## 3. Testing in Postman

### Step 1 — Set Up Auth

**For Jira Server/Data Center (PAT):**
1. Open your request in Postman
2. Go to **Authorization** tab
3. Select Type → **Bearer Token**
4. Paste your Personal Access Token

**For Jira Cloud (API Token):**
1. Go to **Authorization** tab
2. Select Type → **Basic Auth**
3. Username → your email
4. Password → your API token

---

### Step 2 — Set Headers

In the **Headers** tab, add:

| Key | Value |
|---|---|
| `Content-Type` | `application/json` |
| `Accept` | `application/json` |

---

### Step 3 — Test: Get Project Versions

Use this first to confirm your PAT works and to get valid version IDs/names.

- **Method:** `GET`
- **URL:** `https://your-domain.atlassian.net/rest/api/3/project/PROJ/versions`
- **Body:** None

**Sample Response:**
```json
[
  { "id": "10001", "name": "To be Prioritized", "released": false },
  { "id": "10002", "name": "v2.1.0", "released": false }
]
```

---

### Step 4 — Test: Update fixVersion

- **Method:** `PUT`
- **URL:** `https://your-domain.atlassian.net/rest/api/3/issue/PROJ-101`
- **Body (raw → JSON):**

**Set fixVersion (replace all):**
```json
{
  "update": {
    "fixVersions": [{ "set": [{ "name": "To be Prioritized" }] }]
  }
}
```

**Add fixVersion (keep existing):**
```json
{
  "update": {
    "fixVersions": [{ "add": { "name": "To be Prioritized" } }]
  }
}
```

**Set fixVersion by ID (more reliable):**
```json
{
  "update": {
    "fixVersions": [{ "set": [{ "id": "10001" }] }]
  }
}
```

**Expected Response:** `HTTP 204 No Content` (empty body = success)

---

### Step 5 — Find Story Points Field ID

- **Method:** `GET`
- **URL:** `https://your-domain.atlassian.net/rest/api/3/field`
- **Body:** None

Search the response for fields with names like **"Story Points"**, **"Story point estimate"**. Note the `id` value (e.g., `customfield_10016`).

---

### Step 6 — Test: Update Story Points

- **Method:** `PUT`
- **URL:** `https://your-domain.atlassian.net/rest/api/3/issue/PROJ-101`
- **Body (raw → JSON):**

```json
{
  "fields": {
    "customfield_10016": 5
  }
}
```

> Replace `customfield_10016` with the correct field ID from Step 5.

**Expected Response:** `HTTP 204 No Content`

---

### Step 7 — Update Both Fields in One Call

```json
{
  "fields": {
    "customfield_10016": 8
  },
  "update": {
    "fixVersions": [{ "set": [{ "name": "To be Prioritized" }] }]
  }
}
```

---

## 4. Common Story Points Field IDs

| Field ID | Typically Used In |
|---|---|
| `customfield_10016` | Jira Cloud (most common) |
| `customfield_10028` | Jira Cloud next-gen projects |
| `customfield_10004` | Older Jira Server instances |
| `customfield_10006` | Some self-hosted Jira Server setups |

> Always verify using the `/rest/api/3/field` endpoint — never assume.

---

## 5. Python Code

### Setup

```python
import requests
from requests.auth import HTTPBasicAuth
import json

JIRA_BASE_URL = "https://your-domain.atlassian.net"
EMAIL         = "your-email@example.com"
API_TOKEN     = "your-api-token"
PROJECT_KEY   = "PROJ"

auth    = HTTPBasicAuth(EMAIL, API_TOKEN)
headers = {"Accept": "application/json", "Content-Type": "application/json"}
```

> For **Jira Server with PAT**, replace `HTTPBasicAuth` with:
> ```python
> headers["Authorization"] = f"Bearer YOUR_PAT_TOKEN"
> auth = None
> ```

---

### Get Project Versions

```python
def get_project_versions(project_key: str) -> list:
    url = f"{JIRA_BASE_URL}/rest/api/3/project/{project_key}/versions"
    response = requests.get(url, headers=headers, auth=auth)
    response.raise_for_status()
    versions = response.json()
    for v in versions:
        print(f"  ID: {v['id']}  |  Name: {v['name']}  |  Released: {v.get('released', False)}")
    return versions
```

---

### Update fixVersion

```python
def update_fix_version(issue_key: str, version_name: str, operation: str = "set"):
    """
    operation: "set"    → replace all fixVersions
               "add"    → append to existing fixVersions
               "remove" → remove this version
    """
    url = f"{JIRA_BASE_URL}/rest/api/3/issue/{issue_key}"

    if operation == "set":
        payload = {"update": {"fixVersions": [{"set": [{"name": version_name}]}]}}
    elif operation == "add":
        payload = {"update": {"fixVersions": [{"add": {"name": version_name}}]}}
    elif operation == "remove":
        payload = {"update": {"fixVersions": [{"remove": {"name": version_name}}]}}
    else:
        raise ValueError("operation must be 'set', 'add', or 'remove'")

    response = requests.put(url, headers=headers, auth=auth, data=json.dumps(payload))
    if response.status_code == 204:
        print(f"✅ [{issue_key}] fixVersion '{version_name}' updated ({operation})")
    else:
        print(f"❌ Failed: {response.status_code} — {response.text}")
```

---

### Set fixVersion to "To be Prioritized"

```python
# Replace all fixVersions
update_fix_version("PROJ-101", version_name="To be Prioritized", operation="set")

# Append without removing existing versions
update_fix_version("PROJ-101", version_name="To be Prioritized", operation="add")

# Using version ID (more reliable than name)
url = f"{JIRA_BASE_URL}/rest/api/3/issue/PROJ-101"
payload = {"update": {"fixVersions": [{"set": [{"id": "10001"}]}]}}
requests.put(url, headers=headers, auth=auth, data=json.dumps(payload))
```

---

### Update Story Points

```python
def update_story_points(issue_key: str, points: float, custom_field_id: str = "customfield_10016"):
    url     = f"{JIRA_BASE_URL}/rest/api/3/issue/{issue_key}"
    payload = {"fields": {custom_field_id: points}}
    response = requests.put(url, headers=headers, auth=auth, data=json.dumps(payload))
    if response.status_code == 204:
        print(f"✅ [{issue_key}] Story Points set to {points}")
    else:
        print(f"❌ Failed: {response.status_code} — {response.text}")
```

---

### Update Both Fields Together

```python
def update_issue(issue_key: str, version_name: str = None, story_points: float = None,
                 custom_field_id: str = "customfield_10016"):
    url     = f"{JIRA_BASE_URL}/rest/api/3/issue/{issue_key}"
    payload = {"fields": {}, "update": {}}

    if version_name:
        payload["update"]["fixVersions"] = [{"set": [{"name": version_name}]}]
    if story_points is not None:
        payload["fields"][custom_field_id] = story_points

    response = requests.put(url, headers=headers, auth=auth, data=json.dumps(payload))
    if response.status_code == 204:
        print(f"✅ [{issue_key}] Updated — fixVersion: '{version_name}', Story Points: {story_points}")
    else:
        print(f"❌ Failed: {response.status_code} — {response.text}")

# Example
update_issue("PROJ-101", version_name="To be Prioritized", story_points=8)
```

---

### Find Story Points Field ID

```python
def get_story_points_field_id():
    url      = f"{JIRA_BASE_URL}/rest/api/3/field"
    response = requests.get(url, headers=headers, auth=auth)
    response.raise_for_status()
    for f in response.json():
        if any(kw in f["name"].lower() for kw in ["story", "point", "estimate"]):
            print(f"  ID: {f['id']}  |  Name: {f['name']}")
```

---

## 6. Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `401 Unauthorized` | Invalid/expired token | Regenerate your PAT or API token |
| `400 Bad Request` | Wrong version name/ID | Run `get_project_versions()` to verify |
| `Field cannot be set` (Story Points) | Field not on Edit screen | Add Story Points to the Edit Screen in Jira admin |
| `204` with no body | ✅ Success | This is expected — no body means it worked |
| `404 Not Found` | Wrong issue key or URL | Double-check base URL and issue key |

---

## 7. Quick Tips

- **HTTP 204** = success. The PUT endpoint returns no body on success.
- **Always use version ID over name** when possible — names can have spaces/casing issues.
- **Story Points field ID varies** — always call `/rest/api/3/field` to find the correct one for your instance.
- **PAT vs API Token** — PATs are for Jira Server/Data Center. Jira Cloud uses email + API token via Basic Auth.
- Use Postman's **Environment Variables** to store `base_url`, `token`, and `project_key` for reuse across requests.
