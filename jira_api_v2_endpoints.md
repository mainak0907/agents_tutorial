# Jira REST API v2 — Transitions, Fix Versions & Assignee

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

## 1. Update Transition

**Moves an issue to a different workflow status (e.g., To Do → In Progress → Done)**

```
POST /rest/api/2/issue/{issueIdOrKey}/transitions
```

### Example Request

```http
POST https://{your-domain}.atlassian.net/rest/api/2/issue/PROJ-123/transitions
Content-Type: application/json
Authorization: Basic <base64_credentials>
```

### Request Body

```json
{
  "transition": {
    "id": "31"
  }
}
```

### Get Available Transition IDs

```
GET /rest/api/2/issue/{issueIdOrKey}/transitions
```

---

## 2. Update Fix Version

**Sets, adds, or removes fix versions on an issue**

```
PUT /rest/api/2/issue/{issueIdOrKey}
```

### Example Request

```http
PUT https://{your-domain}.atlassian.net/rest/api/2/issue/PROJ-123
Content-Type: application/json
Authorization: Basic <base64_credentials>
```

### Request Body — Set (replaces all existing)

```json
{
  "update": {
    "fixVersions": [
      { "set": [ { "name": "v2.1.0" } ] }
    ]
  }
}
```

### Request Body — Add (keeps existing, adds new)

```json
{
  "update": {
    "fixVersions": [
      { "add": { "name": "v2.1.0" } }
    ]
  }
}
```

### Request Body — Remove (removes a specific version)

```json
{
  "update": {
    "fixVersions": [
      { "remove": { "name": "v1.0.0" } }
    ]
  }
}
```

---

## 3. Update Assignee

**Assigns or unassigns a user on an issue**

```
PUT /rest/api/2/issue/{issueIdOrKey}/assignee
```

### Example Request

```http
PUT https://{your-domain}.atlassian.net/rest/api/2/issue/PROJ-123/assignee
Content-Type: application/json
Authorization: Basic <base64_credentials>
```

### Request Body — Assign a User

```json
{
  "accountId": "5b109f2e9729b51b54dc274d"
}
```

### Request Body — Unassign

```json
{
  "accountId": null
}
```

> **Note:** For Jira Server (older versions), use `"name": "username"` instead of `accountId`.

### Alternative — via General Issue Endpoint

```http
PUT /rest/api/2/issue/{issueIdOrKey}

{
  "fields": {
    "assignee": { "accountId": "5b109f2e9729b51b54dc274d" }
  }
}
```

---

## Quick Reference

| Action                  | Method | Endpoint                                    |
|-------------------------|--------|---------------------------------------------|
| Transition issue        | POST   | `/rest/api/2/issue/{key}/transitions`       |
| Update fix version      | PUT    | `/rest/api/2/issue/{key}`                   |
| Update assignee         | PUT    | `/rest/api/2/issue/{key}/assignee`          |
| Get transitions list    | GET    | `/rest/api/2/issue/{key}/transitions`       |
