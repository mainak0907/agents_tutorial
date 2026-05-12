# Jira REST API v2 — Worklog

## Headers (Common for All Endpoints)

```http
Content-Type: application/json
Authorization: Basic <base64(email:api_token)>
          OR
Authorization: Bearer <personal_access_token>
```

> **Note:** For Jira Cloud, encode `user@example.com:your_api_token` in Base64.  
> For Jira Server/Data Center, a Bearer Personal Access Token (PAT) is also accepted.

---

## Add Worklog to an Issue

```
POST /rest/api/2/issue/{issueIdOrKey}/worklog
```

### Example Request

```http
POST https://{your-domain}.atlassian.net/rest/api/2/issue/PROJ-123/worklog
Content-Type: application/json
Authorization: Basic <base64_credentials>
```

### Request Body

```json
{
  "started": "2026-05-12T10:00:00.000+0000",
  "timeSpent": "3h 30m",
  "comment": "Fixed the login bug and wrote unit tests."
}
```

### Field Details

| Field       | Required | Format                        | Example                          |
|-------------|----------|-------------------------------|----------------------------------|
| `started`   | Yes      | `yyyy-MM-dd'T'HH:mm:ss.SSSZ` | `"2026-05-12T10:00:00.000+0000"` |
| `timeSpent` | Yes      | Jira duration string          | `"3h"`, `"30m"`, `"1h 30m"`, `"1d"` |
| `comment`   | No       | Plain string                  | `"Working on bug fix"`           |

---

### Example Response (201 Created)

```json
{
  "id": "100028",
  "issueId": "10002",
  "author": {
    "accountId": "5b109f2e9729b51b54dc274d",
    "displayName": "John Doe",
    "emailAddress": "john.doe@example.com"
  },
  "comment": "Fixed the login bug and wrote unit tests.",
  "started": "2026-05-12T10:00:00.000+0000",
  "timeSpent": "3h 30m",
  "timeSpentSeconds": 12600
}
```

---

### Using cURL

```bash
curl -u user@example.com:your_api_token \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "started": "2026-05-12T10:00:00.000+0000",
    "timeSpent": "3h 30m",
    "comment": "Fixed the login bug and wrote unit tests."
  }' \
  "https://{your-domain}.atlassian.net/rest/api/2/issue/PROJ-123/worklog"
```

---

## Other Worklog Endpoints

### Get All Worklogs

```
GET /rest/api/2/issue/{issueIdOrKey}/worklog
```

```http
GET https://{your-domain}.atlassian.net/rest/api/2/issue/PROJ-123/worklog
Content-Type: application/json
Authorization: Basic <base64_credentials>
```

---

### Get a Specific Worklog

```
GET /rest/api/2/issue/{issueIdOrKey}/worklog/{worklogId}
```

```http
GET https://{your-domain}.atlassian.net/rest/api/2/issue/PROJ-123/worklog/100028
Content-Type: application/json
Authorization: Basic <base64_credentials>
```

---

### Update a Worklog

```
PUT /rest/api/2/issue/{issueIdOrKey}/worklog/{worklogId}
```

```http
PUT https://{your-domain}.atlassian.net/rest/api/2/issue/PROJ-123/worklog/100028
Content-Type: application/json
Authorization: Basic <base64_credentials>
```

```json
{
  "started": "2026-05-12T14:00:00.000+0000",
  "timeSpent": "5h",
  "comment": "Updated worklog with additional hours."
}
```

---

### Delete a Worklog

```
DELETE /rest/api/2/issue/{issueIdOrKey}/worklog/{worklogId}
```

```http
DELETE https://{your-domain}.atlassian.net/rest/api/2/issue/PROJ-123/worklog/100028
Content-Type: application/json
Authorization: Basic <base64_credentials>
```

---

## Quick Reference

| Action               | Method   | Endpoint                                          |
|----------------------|----------|---------------------------------------------------|
| Add worklog          | `POST`   | `/rest/api/2/issue/{key}/worklog`                 |
| Get all worklogs     | `GET`    | `/rest/api/2/issue/{key}/worklog`                 |
| Get specific worklog | `GET`    | `/rest/api/2/issue/{key}/worklog/{worklogId}`     |
| Update worklog       | `PUT`    | `/rest/api/2/issue/{key}/worklog/{worklogId}`     |
| Delete worklog       | `DELETE` | `/rest/api/2/issue/{key}/worklog/{worklogId}`     |

---

> **Tip:** The `started` date must include the time and timezone offset. Using `+0000` sets it to UTC — adjust the offset to match your timezone (e.g., `+0530` for IST).
