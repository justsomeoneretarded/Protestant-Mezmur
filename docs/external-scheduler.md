# External Scheduler: cron-job.org

This repository is prepared for an external trigger, but saving code does not
create a job in your cron-job.org account. GitHub's built-in cron stays off.
Create **one** external job; do not enable a second scheduler for the same workflow.

## 1. Create a Restricted GitHub Token

Open [the prefilled fine-grained token form](https://github.com/settings/personal-access-tokens/new?name=Rehoboth+Mezmur+Scheduler&target_name=justsomeoneretarded&actions=write&expires_in=90).

- Resource owner: the account that owns Protestant-Mezmur.
- Repository access: **Only select repositories > Protestant-Mezmur**.
- Repository permissions: **Actions > Read and write**. Metadata read access is automatic.
- Use an expiry, then renew and replace the token before it expires.
- Generate the token and put it only in the cron-job.org Authorization header below.
  Do not commit it, put it in a URL/request body, paste it into chat, or expose it in screenshots.
  cron-job.org will hold this credential, so restrict it to this repository.

No GitHub Contents write permission, Telegram credentials, Google OAuth JSON or
cron-job.org API key is needed for this manual console setup.
[GitHub's workflow-dispatch authentication requirements](https://docs.github.com/en/rest/actions/workflows#create-a-workflow-dispatch-event).

## 2. Create the Cron Job

Open [cron-job.org Console](https://console.cron-job.org/) and choose **Create cronjob**.

| Setting | Value |
| --- | --- |
| Title | Rehoboth Mezmur - one song every 3 hours |
| URL | `https://api.github.com/repos/justsomeoneretarded/Protestant-Mezmur/actions/workflows/generate.yml/dispatches` |
| Request method | **POST** (Advanced/request settings) |
| Time zone | **Africa/Nairobi** |
| Schedule | Custom: minute **17**, hours **0, 3, 6, 9, 12, 15, 18, 21**; every day/month/weekday |
| HTTP Basic authentication | Off; use the custom Authorization header below |
| Request timeout | 30 seconds is enough to submit a dispatch; it does not wait for video generation |
| Notifications | Enable failure and automatic-disable notifications |

This is equivalent to `17 */3 * * *` in EAT. The three-hour interval was chosen
from the requested two-to-three-hour range. To use two hours later, select hours
0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22 instead.

## 3. Add These Custom Headers

Use the Advanced/request-headers area. Replace ONLY the token placeholder with
the token generated above; the word `Bearer` and its following space must remain.

```http
Authorization: Bearer YOUR_GITHUB_TOKEN
Accept: application/vnd.github+json
Content-Type: application/json
X-GitHub-Api-Version: 2026-03-10
```

Do not use a cron-job.org API key in this header: this request goes to **GitHub**.

## 4. Set the Request Body

The same non-secret JSON is stored in [cron-dispatch.json](cron-dispatch.json).

```json
{
  "ref": "main",
  "inputs": {
    "batch_size": "1",
    "message_ids": "",
    "upload_youtube": "true"
  }
}
```

This selects the next new song, not a replay, and uses the original recording with
no music rebuilding or sound effects. The workflow selects thumbnail D, with
left-side text and the large right-side circular portrait, for every upload.
If no verified portrait exists, the existing text-only fallback applies.

## 5. Save and Test Once

Save/enable the job, then use **Test run** once. A successful dispatch on the
selected GitHub API version returns HTTP **200** with a workflow-run ID and URL.
Open [Generate Mezmur Videos](https://github.com/justsomeoneretarded/Protestant-Mezmur/actions/workflows/generate.yml)
and confirm a new run appears.

A test dispatch starts a **real one-song generation/upload run**. Its video is
uploaded as Unlisted first, then made Private when its four-hour YouTube publication
slot is applied. It remains Private until the public release time.
The HTTP response only confirms GitHub accepted the request, not that rendering
or uploading finished. cron-job.org cannot monitor the later Actions outcome;
use the Actions page, existing Telegram notices and run summaries for that.
If a request times out, check Actions before testing again: dispatch is not
idempotent and repeated requests can start extra batches.

HTTP 401 usually means an expired/invalid token; 403/404 can mean missing Actions
write permission, wrong repository access or a workflow disabled in GitHub.
HTTP 422 usually means an invalid branch, JSON body or input. Renew the token
when necessary and check the cron-job.org job has not been auto-disabled.

## How the Two Schedules Coexist

- External requests and manual runs use the same `mezmur-generate` concurrency lock.
  Manual upload-only runs share that lock; the publisher called by its own parent
  uses a distinct group to avoid waiting for the lock its parent already holds.
- GitHub allows one active and at most one pending run per group. A newer pending
  run can replace an older pending run; this is not a durable FIFO of every tick.
  Each actual run reads saved Telegram progress. An empty selection skips rendering/upload.
- YouTube still publishes **four hours apart**, using its own native scheduler.
  New batches append after this app's last saved future slot, including earlier
  batches. Existing accepted schedules are not moved.
- At one successful song every three hours, generation can produce eight a day,
  but four-hour publication handles six. The queue can grow; this is not a collision.
- Only this app's saved reservations are tracked. Manual Studio scheduling changes
  and videos uploaded outside this pipeline are not synchronized.
- The external service and GitHub can both experience delays; neither guarantees
  exact-second execution. Token expiry, service outages and Actions/API limits
  can stop triggering. Keep failure notifications enabled.
- Pause the job in cron-job.org to stop automatic generation. An already running
  Actions workflow and videos already scheduled on YouTube are unaffected.

Official references: [cron-job.org request methods and headers](https://cron-job.org/en/faq/),
[GitHub workflow dispatch](https://docs.github.com/en/rest/actions/workflows#create-a-workflow-dispatch-event),
[fine-grained token setup](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens).
